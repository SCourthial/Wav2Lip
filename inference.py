from os import listdir, path, makedirs
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import soundfile as sf

from importlib.machinery import SourceFileLoader

gfpgan = SourceFileLoader("gfpgan","/lip_sync/gfpgan/gfpgan/__init__.py").load_module()

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str,
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str,
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfolder', type=str, help='Folder path to save result. See default for an e.g.',
								default='results')

parser.add_argument('--static', type=bool,
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--batch_size', type=int, help='Batch size for frame buffering', default=128)
parser.add_argument('--face_det_batch_size', type=int,
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=True, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--face_landmarks_detector_path', default='weights/face_landmarker_v2_with_blendshapes.task', type=str,
					help='Path to face landmarks detector')

parser.add_argument('--with_face_mask', default=True, action='store_true',
					help='Blend output into original frame using a face mask rather than directly blending the face box. This prevents a lower resolution square artifact around lower face')

parser.add_argument('--enhance', type=bool, default=True,
					help='Enhance output video with GFPGAN')

args = parser.parse_args()
args.img_size = 96
args.min_batch_size = 32

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	empty_indexes = []
	for idx, box in enumerate(boxes):
		if not box.any(): empty_indexes.append(idx)

	print('Empty boxes: {}'.format(empty_indexes))

	empty_indexes = iter(empty_indexes)
	next_empty_index = next(empty_indexes, None)
	for i in range(len(boxes)):
		if i == next_empty_index:
			next_empty_index = next(empty_indexes, None)
			continue

		if next_empty_index is not None and i + T > next_empty_index:
			window = boxes[i : next_empty_index]
		elif i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size

	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			results.append([0, 0, 0, 0])
		else:
			y1 = max(0, rect[1] - pady1)
			y2 = min(image.shape[0], rect[3] + pady2)
			x1 = max(0, rect[0] - padx1)
			x2 = min(image.shape[1], rect[2] + padx2)

			results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [([image[y1: y2, x1:x2], (y1, y2, x1, x2)] if x1 is not None and y1 is not None and x2 is not None and y2 is not None else [[],[]]) for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		if any(coords):
			face = cv2.resize(face, (args.img_size, args.img_size))
			coords_batch.append(coords)
		else:
			face = np.random.rand(args.img_size, args.img_size, 3) * 255
			coords_batch.append([])

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def read_audio_section(filename, start_time, stop_time):
	track = sf.SoundFile(filename)

	can_seek = track.seekable() # True
	if not can_seek:
		raise ValueError("Not compatible with seeking")

	sr = track.samplerate
	start_frame = round(sr * start_time)
	frames_to_read = round(sr * (stop_time - start_time))
	track.seek(start_frame)
	audio_section = track.read(frames_to_read)
	return audio_section

def face_mask_from_image(image, face_landmarks_detector):
	"""
	Calculate face mask from image. This is done by

	Args:
		image: numpy array of an image
		face_landmarks_detector: mediapipa face landmarks detector
	Returns:
		A uint8 numpy array with the same height and width of the input image, containing a binary mask of the face in the image
	"""
	image_shape = (image.shape[0], image.shape[1])

	# detect face landmarks
	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
	detection = face_landmarks_detector.detect(mp_image)

	mouth_landmarks = [57, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43]
	lower_face_landmarks = [132, 177, 147, 205, 203, 98, 97, 2, 326, 327, 423, 425, 376, 401, 361, 435, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132]
	landmarks_list = [mouth_landmarks, lower_face_landmarks]

	if len(detection.face_landmarks) == 0:
		# no face detected - set mask to all of the image
		return [np.ones(image_shape, dtype=np.uint8) for _ in landmarks_list]

	result = []
	for landmarks in landmarks_list:
		# extract landmarks coordinates
		face_coords = np.array([[detection.face_landmarks[0][idx].x * image.shape[1], detection.face_landmarks[0][idx].y * image.shape[0]] for idx in landmarks])

		# calculate convex hull from face coordinates
		convex_hull = cv2.convexHull(face_coords.astype(np.float32))

		# apply convex hull to mask
		result.append(cv2.fillPoly(np.zeros(image_shape, dtype=np.uint8), pts=[convex_hull.squeeze().astype(np.int32)], color=1))

	return result

def read_next_video_frames(video_stream, frame_count):
	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			return False, frames

		if len(frames) >= frame_count:
			return True, frames

		if args.resize_factor > 1:
			frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

		if args.rotate:
			frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

		y1, y2, x1, x2 = args.crop
		if x2 == -1: x2 = frame.shape[1]
		if y2 == -1: y2 = frame.shape[0]

		frame = frame[y1:y2, x1:x2]

		frames.append(frame)

def inference(full_frames, start_time=0, stop_time=None, index_offset=0, face_landmarks_detector=None, model=None, restorer=None):
	print ("Number of frames available for inference: "+str(len(full_frames)))

	wav = read_audio_section(args.audio, start_time, stop_time)
	mel = audio.melspectrogram(wav)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_step_size = min(16, len(mel[0])) if len(full_frames) > 1 else len(mel[0]) + 1
	mel_idx_multiplier = max(len(mel[0]) - mel_step_size, 1) / ((len(full_frames)) - 2) if len(full_frames) > 2 else 1
	print('mel_idx_multiplier: {}'.format(mel_idx_multiplier))
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	if len(full_frames) > len(mel_chunks):
		for i in range(len(full_frames) - (len(mel_chunks))):
			mel_chunks.append(np.zeros(mel_chunks[0].shape))

		print("New length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	os.makedirs(args.outfolder, exist_ok=True)

	new_index_offset = index_offset
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for j, (p, f, c) in enumerate(zip(pred, frames, coords)):
			if len(c) > 0:
				y1, y2, x1, x2 = c

				p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

				if args.enhance:
					_, _, p = restorer.enhance(
						p,
						has_aligned=False,
						only_center_face=True,
						paste_back=True,
						weight=0.5
					)

				if face_landmarks_detector:
					raw_mouth_mask, raw_lower_face_mask = face_mask_from_image(p, face_landmarks_detector)

					edge_kernel = np.ones((20, 20), np.uint8)
					raw_edge_mask = cv2.erode(raw_lower_face_mask, edge_kernel, iterations=1)

					edge_mask = (raw_lower_face_mask - raw_edge_mask)[..., None]
					lower_face_mask = (raw_edge_mask - raw_mouth_mask)[..., None]
					mouth_mask = raw_mouth_mask[..., None]

					p = f[y1:y2, x1:x2] * (1 - raw_lower_face_mask[..., None]) \
						+ (f[y1:y2, x1:x2] * edge_mask * 0.6 + p * edge_mask * 0.4) \
						+ (f[y1:y2, x1:x2] * lower_face_mask * 0.3 + p * lower_face_mask * 0.7) \
						+ p * mouth_mask

					contour_kernel_size = min(y2 - y1, x2 - x1, 150) // 50
					if contour_kernel_size > 0:
						new_face_blurred = cv2.GaussianBlur(p, (7, 7), 0)

						contour_kernel = np.ones((contour_kernel_size, contour_kernel_size), np.uint8)
						outer_edge_contours, _ = cv2.findContours(cv2.dilate(raw_edge_mask, contour_kernel, iterations=1), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
						inner_edge_contours, _ = cv2.findContours(cv2.erode(raw_edge_mask, contour_kernel, iterations=1), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
						outer_lower_face_contours, _ = cv2.findContours(cv2.dilate(raw_lower_face_mask, contour_kernel, iterations=1), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
						inner_lower_face_contours, _ = cv2.findContours(cv2.erode(raw_lower_face_mask, contour_kernel, iterations=1), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
						outer_mouth_contours, _ = cv2.findContours(cv2.dilate(raw_mouth_mask, contour_kernel, iterations=1), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
						inner_mouth_contours, _ = cv2.findContours(cv2.erode(raw_mouth_mask, contour_kernel, iterations=1), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

						edge_contour_mask = np.zeros_like(p)
						cv2.drawContours(edge_contour_mask, outer_edge_contours, -1, (255, 255, 255), cv2.FILLED)
						cv2.drawContours(edge_contour_mask, inner_edge_contours, -1, (0, 0, 0), cv2.FILLED)
						lower_face_contour_mask = np.zeros_like(p)
						cv2.drawContours(lower_face_contour_mask, outer_lower_face_contours, -1, (255, 255, 255), cv2.FILLED)
						cv2.drawContours(lower_face_contour_mask, inner_lower_face_contours, -1, (0, 0, 0), cv2.FILLED)
						mouth_contour_mask = np.zeros_like(p)
						cv2.drawContours(mouth_contour_mask, outer_mouth_contours, -1, (255, 255, 255), cv2.FILLED)
						cv2.drawContours(mouth_contour_mask, inner_mouth_contours, -1, (0, 0, 0), cv2.FILLED)

						edge_contour_mask = np.where((edge_contour_mask - lower_face_contour_mask - mouth_contour_mask) > 0, 1, 0)
						lower_face_contour_mask = np.where((lower_face_contour_mask - mouth_contour_mask - edge_contour_mask) > 0, 1, 0)
						mouth_contour_mask = np.where((mouth_contour_mask - edge_contour_mask - lower_face_contour_mask) > 0, 1, 0)

						p = p * (1 - (edge_contour_mask + lower_face_contour_mask + mouth_contour_mask)) \
							+ new_face_blurred * edge_contour_mask \
							+ new_face_blurred * lower_face_contour_mask \
							+ new_face_blurred * mouth_contour_mask

				f[y1:y2, x1:x2] = p

			cv2.imwrite(f'{args.outfolder}/{index_offset + i * batch_size + j:06d}.png', f)
			new_index_offset += 1

	return new_index_offset

def main():
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	face_landmarks_detector = None
	if args.with_face_mask and args.face_landmarks_detector_path:
		base_options = python.BaseOptions(model_asset_path=args.face_landmarks_detector_path, delegate='GPU')
		options = vision.FaceLandmarkerOptions(
			base_options=base_options,
			output_face_blendshapes=True,
			output_facial_transformation_matrixes=True,
			num_faces=1
		)
		face_landmarks_detector = vision.FaceLandmarker.create_from_options(options)

	model = load_model(args.checkpoint_path)
	if args.enhance:
		restorer = gfpgan.GFPGANer(
			model_path='/lip_sync/gfpgan/experiments/pretrained_models/GFPGANv1.4.pth',
			upscale=1,
			arch='clean',
			channel_multiplier=2,
			bg_upsampler=None
		)

	print ("Models loaded")

	if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps

		inference(
			full_frames,
			face_landmarks_detector=face_landmarks_detector,
			model=model,
			restorer=restorer
		)
	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)
		length = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
		start_time = 0.0
		index_offset = 0

		print('Reading video frames...')

		read_frames_count = 0
		while 1:
			remaining_frames_in_next_batch = length - read_frames_count - args.batch_size
			frame_count = length - read_frames_count // 2 if remaining_frames_in_next_batch > 0 and remaining_frames_in_next_batch < args.min_batch_size else args.batch_size
			still_reading, frames = read_next_video_frames(video_stream, frame_count)
			read_frames_count += len(frames)

			stop_time = float(read_frames_count) / fps

			index_offset = inference(
				frames,
				start_time=start_time,
				stop_time=stop_time,
				index_offset=index_offset,
				face_landmarks_detector=face_landmarks_detector,
				model=model,
				restorer=restorer
			)

			if not still_reading:
				video_stream.release()
				break

			start_time = stop_time

if __name__ == '__main__':
	main()
