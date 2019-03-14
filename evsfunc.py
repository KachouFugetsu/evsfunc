from vapoursynth import core
import numpy as np
import vapoursynth as vs
import sgvsfunc as sgf
import mvsfunc as mvf
import math
import random

def banding_check(clip, range = 16, y = 64, cb = 64, cr = 64, plane = 0):
	""" 
	Compare the difference between source and debanded clip
	range, y, cb, cr: arguments of f3kdb.
	plane: select which plane to show.
	"""
	
	dbed = core.f3kdb.Deband(clip, range, y, cb, cr, preset = "nograin", output_depth = clip.format.bits_per_sample, dither_algo = 3)
	diff = core.std.MakeDiff(clip, dbed)
	gain = 2 ** (clip.format.bits_per_sample - 7)
	mid = 2 ** (clip.format.bits_per_sample - 1)
	peak = 2 ** clip.format.bits_per_sample - 1
	expr = "x "+str(mid)+" - abs "+str(gain)+" *"
	diff = core.std.Expr(diff, expr)
	diff = core.std.ShufflePlanes(diff, plane, vs.GRAY)
	return diff

def FrameInfoMod(clip, title):
	
	""" cosmetic change from FrameInfo in sgvsfunc """
	
	def FrameProps(n, clip):
		clip = core.sub.Subtitle(clip, "Frame Number: " + str(n) + " of " + str(clip.num_frames) + "\nPicture Type: " + clip.get_frame(n).props._PictType.decode(), style = "Courier New,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,7,10,10,10,1", fontdir="D:/fonts")
		return clip

	clip = core.std.FrameEval(clip, functools.partial(FrameProps, clip=clip))
	clip = core.sub.Subtitle(clip, text=[title], style = "Courier New,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,9,10,10,10,1", fontdir="D:/fonts")

	return clip

def tm(c, display = 120, maxcll = 4000, tmo = "Mobius", format = vs.RGB24, dither = "error_diffusion", chromaloc = 2, show_plane = -1):

	"""
	A wrapper function for tonemapping UHD HDR contents to LDR display.
	display:	target display peak brightness. (Default: 120)
	maxcll:		Max content luminance level. (Default: 4000)
	tmo:		tonemap operator used (Default: Mobius)
				Hable: Preserve detail better, but overall darker so by default gain is multiplied by 2
				Reinhard: Better for color accuracy, highlights easily get clipped
				Mobius: Similar to Reinhard, better contrast
	format:		Output format (Default: vs.RGB24)
				vs.RGB24 (recommended for higher display precision)
				vs.YUV420P8 (if you need to check YUV values)
	dither:		Same as the dithering method specified in core.resize (Default: "error_diffusion")
	chromaloc:	Same as the chroma location specified in core.resize (Default: 2)
	show_plane:	Whether to show individual plane of output clip. Only works for YUV output. (Default: -1)
				-1: don't show individual planes.
				0~2: Show corresponding plane.
	"""	
	
	gain = 1.2 * maxcll / display
	
	# sanity check
	if format == vs.RGB24:
		matrix_o = 0
	elif format == vs.YUV420P8:
		matrix_o = 1
	else:
		raise TypeError("Only YUV420P8 and RGB24 are supported for output format.")
		
	# convert to RGB
	c = core.resize.Bicubic(c, format = vs.RGBS, range_in_s = "limited", matrix_in_s = "2020ncl", primaries_in_s = "2020", primaries_s = "2020", transfer_in_s = "st2084", transfer_s = "linear", chromaloc_in = chromaloc, nominal_luminance = maxcll, filter_param_a = 0, filter_param_b = 0.5, dither_type = "none")	
	# tonemap with selected TMO
	if tmo is "Hable":
		c = core.tonemap.Hable(c, 2*gain)
	elif tmo is "Reinhard":
		c = core.tonemap.Reinhard(c, gain, peak = maxcll)
	else:
		c = core.tonemap.Mobius(c, gain, peak = maxcll)
	# convert to output format		
	c = core.resize.Bicubic(c, format = format, matrix = matrix_o, primaries_in_s = "2020", primaries_s = "709", transfer_in_s = "linear", transfer_s = "709", dither_type = dither, nominal_luminance = display, filter_param_a = 0, filter_param_b = 0.5)
	
	if (show_plane > -1 and format == vs.YUV420P8):		
		return core.std.ShufflePlanes(c, show_plane, vs.GRAY)
	else:
		return c

def tm_hybrid(c, display = 120, maxcll = 4000):

	"""
	Simple weighted result of Mobius and Hable tonemapping trying to achieve accurate color in dark and detail preservation in bright area.
	"""
	gain = 1.2*maxcll/display
	c = core.resize.Bicubic(c, format = vs.RGBS, range_in_s = "limited", matrix_in_s = "2020ncl", primaries_in_s = "2020", primaries_s = "2020", transfer_in_s = "st2084", transfer_s = "linear", chromaloc_in = 2, nominal_luminance = maxcll, filter_param_a = 0, filter_param_b = 0.5, dither_type = "none")	
	c1 = core.tonemap.Mobius(c, gain, peak = maxcll)
	c1 = core.resize.Bicubic(c1, format = vs.YUV444P16, matrix_s = "709", primaries_in_s = "2020", primaries_s = "709", transfer_in_s = "linear", transfer_s = "709", dither_type = "error_diffusion", nominal_luminance = display, filter_param_a = 0, filter_param_b = 0.5)
	c2 = core.tonemap.Hable(c, 2*gain)
	c2 = core.resize.Bicubic(c2, format = vs.YUV444P16, matrix_s = "709", primaries_in_s = "2020", primaries_s = "709", transfer_in_s = "linear", transfer_s = "709", dither_type = "error_diffusion", nominal_luminance = display, filter_param_a = 0, filter_param_b = 0.5)
	c = core.std.MaskedMerge(c1, c2, c2, first_plane = True)
	c = core.resize.Bicubic(c, format = vs.RGB24, matrix_in_s = "709", primaries_in_s = "709", primaries_s = "709", transfer_in_s = "709", transfer_s = "709", dither_type = "error_diffusion", nominal_luminance = display)
	return c
	
	
def tonemap(src, light_s, light_d):
	gain = 5*light_s/light_d
	src = core.resize.Bicubic(clip = src, format = vs.RGBS, filter_param_a = 0, filter_param_b = 0.75, range_in_s = "limited", matrix_in_s = "2020ncl", primaries_in_s = "2020", transfer_in_s = "st2084", transfer_s = "linear", dither_type = "none", nominal_luminance = light_s)
	src = core.tonemap.Hable(src, gain)
	src = core.resize.Bicubic(clip = src, format = vs.YUV420P16, filter_param_a = 0, filter_param_b = 0.75, primaries_in_s = "2020", transfer_in_s = "linear", matrix_s = "709", primaries_s = "709", transfer_s = "709", dither_type = "none")
	return src
	
def nr_f3kdb(src, range = 16, y = 64, cb = 64, cr = 64, grainy = 0, grainc = 0, dyn_grain = False, keep_tv_range = True, thr = 0.35, thrc = None, elast = 2.5, output_depth = 16):
	if thrc == None:  thrc = thr
	nr = core.rgvs.RemoveGrain(src, 20)
	diff = core.std.MakeDiff(src, nr)
	db = core.f3kdb.Deband(nr, range, y, cb, cr, grainy, grainc, keep_tv_range = keep_tv_range, output_depth = output_depth)
	db = core.std.MergeDiff(db, diff)
	db = mvf.LimitFilter(db, src, thr = thr, thrc = thrc, elast = elast)
	return db
	
def gaussian_usm(src, sigma = 1.0, sigmaV = None, thr = 1.0, thrc = None, elast = 3.0):
	
	if sigmaV == None: sigmaV = sigma
	if thrc == None:  thrc = thr
	
	blur = core.bilateral.Gaussian(src, sigma, sigmaV)
	diff = core.std.MakeDiff(src, blur)
	sharp = core.std.MergeDiff(src, diff)
	
	res = mvf.LimitFilter(sharp, src, thr = thr, thrc = thrc, elast = elast)
	
	return res

def mask_inflate(src, mode = "inflate", n = 1):
	"""
	Simple mask expand/inpand wrapper.
	src: mask clip to process.
	mode: 	"inflate"
			"expand" (dilation)
			"deflate"
			"depand" (erosion)
	n: number of passes
	"""
	
	if mode is "inflate":
		for n in range(0, n):
			src = core.std.Inflate(src)
	elif mode is "expand" or "maximum" or "dilation":
		for n in range(0, n):
			src = core.std.Maximum(src)
	elif mode is "deflate":
		for n in range(0, n):
			src = core.std.Deflate(src)
	elif mode is "depand" or "minimum" or "erosion":
		for n in range(0, n):
			src = core.std.Minimum(src)
	else:
		raise TypeError("mode is not supported")
	
	return src

def random_compare(clips, seed1 = 33, seed2 = 66):
	random.seed(seed1)
	offs = random.randint(0,10000)*2
	random.seed(seed2)
	every = random.randint(10000,20000)*2
	cpr = core.std.Interleave(clips)
	cpr = sgf.SelectRangeEvery(cpr, offset = offs, every = every, length = 2)
	cpr = mvf.ToRGB(cpr, depth = 8, dither = 6, full = False, matrix = "709")
	return cpr

def save_screens(clip, count):
	"""
	Generate short random comparison clip.
	clip: input comparison clip, to function properly, it needs to be interleaved with source and encode clip and converted to RGB24.
	count: number of groups of comparison.
	"""

	index = np.random.randint(clip.num_frames/2, size = count)
	out_index = []
	for n in index:
		while clip.get_frame(2*n+1).props._PictType.decode() is not "B":
			n = n + 1
		out_index.append(2*n)
		out_index.append(2*n+1)
		
	out_clip = 0
	for n in out_index:
		if out_clip is 0:
			out_clip = core.std.Trim(clip, n, length = 1)
		else:
			out_clip = out_clip+core.std.Trim(clip, n, length = 1)
	
	return out_clip