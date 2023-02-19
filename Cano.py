#!/usr/bin/env python
import sys
import os
import logging
from skimage.transform import rotate, warp
import numpy as np
import skimage.io as io
import argparse
from tqdm import tqdm
import multiprocessing
from pprint import pformat
import csv
from distutils.util import strtobool


class EmptyDirError(ValueError):
    pass


# Custom log formatter
class CustomFormatter(logging.Formatter):
    grey = "\033[37m"
    cyan = "\033[96m"
    yellow = "\033[93m"
    red = "\033[31m"
    bold_red = "\033[91;1m"
    reset = "\033[0m"
    format = "%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: cyan + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("lai_finder")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


# Start main functions

def fisheye_in_polar(coords, output_shape):
    """Convert a fisheye image to polar coordinates"""
    x = coords[:, 0]
    y = coords[:, 1]
    x_center = output_shape[0]/2
    y_center = x_center
    #  dist from center, sqrt2 accounts for square shape. where >1 is undefined
    r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    r = r / max(r) * np.sqrt(2)
    # polar angle, theta, with origin set to the center
    y_invert_centered = y[::-1] - y_center
    x_centered = x - x_center
    theta = np.arctan2(y_invert_centered, x_centered)
    # scale theta between 0 and 1
    theta[theta < 0] += 2*np.pi
    theta /= 2*np.pi
    return np.vstack((r, theta)).T


def polar_to_equidistant(r_theta, input_shape):
    """Convert a set of polar coordinates to equidistant (hemispherical coords)"""
    r, theta = r_theta[:, 0], r_theta[:, 1]
    max_x, max_y = input_shape[1]-1, input_shape[0]-1
    xs = theta * max_x
    ys = r * max_y
    return np.vstack((xs, ys)).T


def inverse_map(fisheye_xy, output_shape, input_shape):
    """Map cylindrical image to hemispherical projection"""
    polar = fisheye_in_polar(fisheye_xy, output_shape)
    equi_xy = polar_to_equidistant(polar, input_shape)
    return equi_xy


def image2hemiphot(hemi):
    """Infer shape parameters from hemi"""
    cy, cx, _ = hemi.shape
    cy = cy / 2  # y center
    cx = cx / 2  # x center
    cr = cy - 2  # radius
    return cx, cy, cr


def threshold_image(im_hemi, threshold=0.82):
    """Threshold hemispherical image by a given threshold"""
    im_hemi[im_hemi > threshold] = 1
    im_hemi[im_hemi <= threshold] = 0
    return im_hemi


def calc_gap_fractions(im_segment, circ_params):
    """Calculate gap fractions across 89 altitudinal circles"""
    deg2rad = np.pi/180.0
    steps = np.arange(360) + 1
    cx, cy, cr = circ_params
    circles = np.arange(89) + 1
    gap_fractions = np.zeros(89)

    for i in circles:
        x = np.round(cx + np.cos(steps * deg2rad) * i * cr/90., 0)
        y = np.round(cy + np.sin(steps * deg2rad) * i * cr/90., 0)
        for j in steps:
            gap_fractions[i-1] = gap_fractions[i-1] + im_segment[int(y[j-1]) - 1, int(x[j-1]) - 1]

    return np.array(gap_fractions) / 360.0


def calc_lai(gap_fractions, width=6):
    """Calculate LAI as in Hemiphot.R"""
    # angles of LAI2000
    # weights given by Licor canopy analyzer manual
    deg2rad = np.pi/180.0
    angle = np.array([7, 23, 38, 53, 68])
    wi = np.array([0.034, 0.104, 0.160, 0.218, 0.494])
    t = np.zeros(5)

    for i in range(-6, 7):    # Fixed from original, see Issue #17
        angle_idx = angle + i - 1
        t += gap_fractions[angle_idx]

    t /= (2 * width + 1)
    return 2 * np.sum(-np.log(t) * wi * np.cos(angle*deg2rad))


def transform_image(imgpath, slicepoint=2176, rotate_deg=-90):
    """Crop, warp and rotate image"""
    image = io.imread(imgpath)
    pano = image[:slicepoint, :, :]
    input_shape = pano.shape
    output_shape = (input_shape[0] * 2, input_shape[0] * 2)

    polar = warp(pano, inverse_map,
                 map_args={"input_shape": input_shape, "output_shape": output_shape},
                 output_shape=output_shape)

    polar = rotate(polar, rotate_deg)   # Rotate is only necessary to exactly replicate original method

    return polar


def threshold_and_lai(polar, threshold=0.82):
    """Threshold image and calculate LAI"""
    # Threshold image
    cx, cy, cr = image2hemiphot(polar)
    blue = polar[:, :, 2]  # Use blue channel as in Hemispherical_2.0
    thresh = threshold_image(np.copy(blue), threshold)  # TODO: May not need to take a copy, could possibly thus save memory

    # Calculate LAI
    gap_fractions = calc_gap_fractions(thresh, (cx, cy, cr))
    lai = calc_lai(gap_fractions)

    return thresh, lai


def process_image_single(imgpath, threshold=0.82, slicepoint=2176, rotate_deg=-90, mode="full", save_files=False, outpath=None, fileext="png"):
    """Process a single image"""
    filename = os.path.splitext(os.path.basename(imgpath))[0]
    # Note: this command is currently brittle if imported into other python scripts.
    # If save_files is True and outpath is not None, saving will probably fail.
    polarfile = f"{filename}_polar.{fileext}"
    threshfile = f"{filename}_thresh.{fileext}"

    if save_files:
        polarfile = os.path.join(outpath, "polar", polarfile)
        threshfile = os.path.join(outpath, "thresh", threshfile)

    # Process
    if mode == "full":
        polar = transform_image(imgpath, slicepoint, rotate_deg)
        thresh, lai = threshold_and_lai(polar, threshold)

        if save_files:
            polar = (polar * 255).astype('uint8')
            io.imsave(polarfile, polar)
            thresh = (thresh * 255).astype('uint8')
            io.imsave(threshfile, thresh)
        return imgpath, lai

    elif mode == "midpoint":
        polar = transform_image(imgpath, slicepoint, rotate_deg)
        if save_files:
            polar = (polar * 255).astype('uint8')
            io.imsave(polarfile, polar)
        return imgpath, None

    elif mode == "pickup":
        # load imgpath
        polar = io.imread(imgpath) / 255.0
        thresh, lai = threshold_and_lai(polar, threshold)
        if save_files:
            thresh = (thresh * 255).astype('uint8')
            io.imsave(threshfile, thresh)
        return imgpath, lai

    else:
        raise ValueError('"mode" argument must be one of "full", "midpoint", or "pickup"')


def process_image_batch(imgpath, threshold=0.82, slicepoint=2176, rotate_deg=-90, mode="full", save_files=False, outpath=None, fileext="png"):
    """Batch process multiple images in single-core mode"""
    # Add any other extensions here later if necessary
    imagepath_list = [os.path.join(imgpath, file) for file in os.listdir(imgpath) if file.endswith((".jpg", ".JPG", ".jpeg", ".JPEG", ".png"))]
    logger.info(f"Processing the following files:\n{pformat(imagepath_list)}")
    if len(imagepath_list) < 1:
        raise EmptyDirError(f"{imgpath} has no valid files inside it! Are you sure it's the right path?")
    # Process list of files
    outlist = [process_image_single(x, threshold=threshold, slicepoint=slicepoint, save_files=save_files, mode=mode, outpath=outpath, fileext=fileext) for x in tqdm(imagepath_list, leave=False)]
    imgpaths, lais = zip(*outlist)
    return imgpaths, lais


def multiprocess_image_batch(imgpath, threshold=0.82, slicepoint=2176, rotate_deg=-90, mode="full", save_files=False, outpath=None, fileext="png", cores=15):
    """Batch process multiple images in multi-core mode"""
    # Add any other extensions here later if necessary
    # This mode is much faster when number of files is large! But it is also very intensive.
    cpus = multiprocessing.cpu_count() - 1
    if cores == 0:
        raise ValueError("Core count cannot be 0!")
    elif cores == 1:
        logger.warning("Only 1 core specified, defaulting to standard batch processing...")
        return process_image_batch(imgpath, threshold, slicepoint, rotate_deg, mode, save_files, outpath, fileext)
    elif cores > 0:
        if cores <= cpus:
            cpus = min(cpus, cores)
        else:
            logger.warning(f"More cores ({cores}) requested than system can provide ({cpus}).")

    # Find files to process
    imagepath_list = [os.path.join(imgpath, file) for file in os.listdir(imgpath) if file.endswith((".jpg", ".JPG", ".jpeg", ".JPEG", ".png"))]

    # Construct args for starmap
    var_list = [[x, threshold, slicepoint, rotate_deg, mode, save_files, outpath, fileext] for x in imagepath_list]
    logger.info(f"Multiprocessing the following files:\n{pformat(imagepath_list)}\non \033[93m{cpus} cores")
    if len(imagepath_list) < 1:
        raise EmptyDirError(f"{imgpath} has no valid files inside it! Are you sure it's the right path?")

    # Starmap single image function onto var list
    outlist = []
    with multiprocessing.Pool(cpus) as p:
        try:
            for result in p.starmap(process_image_single, var_list):
                logger.debug("Completed %s", result[0])
                outlist.append(result)
        except KeyboardInterrupt:
            logger.warning("Attempting to exit multicore run...")
            p.terminate()

    imgpaths, lais = zip(*outlist)
    return imgpaths, lais


def write_results_csv(imgpaths, lais, outpath):
    """Construct, format, and write results.csv"""
    logger.info(f"Writing data to {os.path.join(outpath, 'results.csv')}")
    uids = list(range(1, len(imgpaths)+1))
    imgnames = [os.path.basename(image) for image in imgpaths]
    imgpaths = [os.path.abspath(image) for image in imgpaths]
    contents = list(zip(uids, imgnames, imgpaths, lais))
    contents[:0] = [("uid", "image_name", "image_path", "lai")]
    with open(os.path.join(outpath, "results.csv"), "w", newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(contents)


def print_citations():
    """Print relevant citations"""
    print("\033[31m\nCano.py is a wrapper and CLI for a multicore adaptation of a previous\ndigital hemispherical photography (DHP) analysis pipeline.")
    print("\033[96m\nThis pipeline underlies the code of https://app.cano.fi/\noriginally by Jon Atherton (University of Helsinki)\n"
          "\033[93mhttps://www.doi.org/10.5281/zenodo.5171970\nhttps://github.com/joathert/canofi-app\n")
    print("\033[96mThe LAI inference is based upon Hemiphot.R:\n"
          "\033[93mHans ter Steege (2018)\nHemiphot.R: Free R scripts to analyse hemispherical photographs for canopy openness,\n"
          "leaf area index and photosynthetic active radiation under forest canopies.")
    print("Unpublished report. Naturalis Biodiversity Center, Leiden, The Netherlands")
    print("https://github.com/Hans-ter-Steege/Hemiphot\n")
    print("\033[96mCLI, multiprocessing, optimisation, and extra programming\nby Francis Windram, Imperial College London\033[0m\n")


def main(args):
    """Process arguments, setup, and run script"""
    # Are we in debug mode?
    if args["debug"]:
        logger.setLevel(logging.DEBUG)
        logger.warning("Running in debug mode...")
        logger.debug("Args: \n%s", pformat(args))

    batchmode = False
    # Are we in batch mode? Assume so if "image" is a dir
    if os.path.isdir(args["image"]):
        batchmode = True

    # Work out which mode we are in for processing
    processmode = "full"
    if args["midpoint"]:
        processmode = "midpoint"
        logger.info("Performing polar reprojection only.")
    elif args["pickup"]:
        processmode = "pickup"
        logger.info("Performing thresholding and LAI calculation only.")
    else:
        logger.info("Performing full analysis.")

    # Create output directories
    outdir = None
    resultsdir = None
    if args["save_files"] or args["save_csv"]:
        if args["outdir"]:
            outdir = args["outdir"]
        else:
            # Construct outdir from image location or directory location
            if batchmode:
                outdir = os.path.split(os.path.abspath(args["image"]))[0]
            else:
                outdir = os.path.split(os.path.split(os.path.abspath(args["image"]))[0])[0]
        if os.path.split(outdir)[1] == "results":
            # Detect if we're already in resultsdir
            resultsdir = outdir
        else:
            resultsdir = os.path.join(outdir, "results")
        try:
            os.mkdir(resultsdir)
            logger.info("Made results directory at %s", resultsdir)
        except FileExistsError:
            logger.warning("Results directory already exists at %s\nSome files may be overwritten.", resultsdir)

        # Make results paths if they don't exist and we're saving files.
        logger.info("Making results subdirs...")
        if processmode in ["full", "midpoint"]:
            try:
                os.mkdir(os.path.join(resultsdir, "polar"))
            except FileExistsError:
                logger.debug("Polar folder already exists.")
        if processmode in ["full", "pickup"]:
            try:
                os.mkdir(os.path.join(resultsdir, "thresh"))
            except FileExistsError:
                logger.debug("Threshold folder already exists.")

    # Do the actual processing!
    try:
        imgpath_out = []
        lai_out = []
        if batchmode:
            logger.info("Running in batch mode.")
            if args["multicore"]:
                imgpath_out, lai_out = multiprocess_image_batch(
                    args["image"],
                    threshold=args["threshold"], slicepoint=args["slice"],
                    mode=processmode, save_files=args["save_files"], outpath=resultsdir, fileext=args["extension"],
                    cores=args["multicore"]
                )
            else:
                imgpath_out, lai_out = process_image_batch(
                    args["image"],
                    threshold=args["threshold"], slicepoint=args["slice"],
                    mode=processmode, save_files=args["save_files"], outpath=resultsdir, fileext=args["extension"]
                )
            logger.info(f"Batch processing complete.")

            if args["save_csv"] and processmode != "midpoint":
                write_results_csv(imgpath_out, lai_out, resultsdir)
        else:
            logger.info("Running in single image mode.")
            imgpath_out, lai_out = process_image_single(
                args["image"],
                threshold=args["threshold"], slicepoint=args["slice"],
                mode=processmode, save_files=args["save_files"], outpath=resultsdir, fileext=args["extension"]
            )
            if lai_out is not None:
                logger.info(f"LAI: {lai_out}")
        logger.info(f"Image processing complete.")
    except KeyboardInterrupt:
        print()
        logger.warning("Exiting...\n\n")
    except Exception:
        logger.exception("Fatal error during processing")
        sys.exit(1)


def testbatch(args, coremin=1, coremax=4, repeats=5, burnin=True):
    if coremin < 1:
        logger.critical("Minimum cores is fewer than 1!")
        raise ValueError(f"Minimum cores = {coremin}!")
    if repeats < 1:
        logger.critical("Must do at least 1 repeat")
        raise ValueError(f"Repeats = {repeats}!")
    elif repeats > 9:
        logger.warning("Large number of repeats specified %i, this could take a long time!", repeats)

    logger.info(f"Starting tests of {repeats} repeats between {coremin} and {coremax} cores")
    from time import time
    coretests = list(range(coremin, coremax+1))
    coretests.reverse()

    processmode = "full"
    if args["midpoint"]:
        processmode = "midpoint"
        logger.info("Performing polar reprojection only.")
    elif args["pickup"]:
        processmode = "pickup"
        logger.info("Performing thresholding and LAI calculation only.")
    else:
        logger.info("Performing full analysis.")

    resultsdir = None

    if burnin:
        logger.info(f"Performing memory burnin on {coremax} cores")
        imgpath_out, lai_out = multiprocess_image_batch(
            args["image"],
            threshold=args["threshold"], slicepoint=args["slice"],
            mode=processmode, save_files=False, outpath=resultsdir,
            cores=coremax
        )
        logger.info(f"Burnin complete, starting timed runs...\n")
    teststart = time()

    test_means = []
    test_sds = []
    logger.disabled = True
    coretestbar = tqdm(coretests)
    for c in coretestbar:
        coretestbar.set_description(f"Cores = {c}")
        times = []
        for i in tqdm(range(repeats), leave=False):
            start = time()
            imgpath_out, lai_out = multiprocess_image_batch(
                args["image"],
                threshold=args["threshold"], slicepoint=args["slice"],
                mode=processmode, save_files=False, outpath=resultsdir,
                cores=c
            )
            end = time()
            times.append(end - start)
        test_means.append(np.mean(times))
        test_sds.append(np.std(times))
    testend = time()
    per_iteration = list(np.array(test_means) / 11)
    result = list(zip(coretests, test_means, test_sds, per_iteration))
    result[:0] = [("Cores", "Mean", "SD", "Per Iteration")]
    logger.disabled = False
    print()
    logger.info(f"Timed runs completed in {testend - teststart}s, results:")
    print(pformat(result))


if __name__ == "__main__":
    # Hacky way to make --citation work without requiring arguments
    if "--citation" in sys.argv:
        print_citations()
        sys.exit(0)

    # Parse arguments appropriately
    parser = argparse.ArgumentParser(description="Transform, threshold, and calculate LAI for panoramic canopy photos.")
    parser.add_argument("image", help="an image to analyse (or folder containing said images)")

    parser.add_argument("-o", "--outdir", nargs="?", help="output directory", metavar="d")
    parser.add_argument("-e", "--extension", nargs="?", help="output file extension", metavar="ext", choices=["png", "jpg"], const="png", default="png")

    midargs = parser.add_mutually_exclusive_group()
    midargs.add_argument("-m", "--midpoint", action="store_true", help="output polar image for standardisation (cannot be combined with -p)")
    midargs.add_argument("-p", "--pickup", action="store_true", help="pick up from standardised polar images for thresholding and LAI calculation (cannot be combined with -m)")
    # NOTE: Running in midpoint/pickup mode causes some error in LAI measurements due to colourspace conversions
    # This is not present in full runs, but doing this forgoes the option for manual standardisation.

    parser.add_argument("-c", "--multicore", nargs="?", type=int, const=-1, help="enable multicore processing", metavar="int")
    parser.add_argument("-d", "--debug", action="store_true", help="enable debugging information")

    paramgroup = parser.add_argument_group("processing parameters")
    paramgroup.add_argument("-t", "--threshold", nargs="?", type=float, const=0.82, default=0.82,
                            help="threshold proportion for LAI calculation (defaults to 0.82)", metavar="flt")
    paramgroup.add_argument("-s", "--slice", nargs="?", type=int, const=2176, default=2176,
                            help="slice point for image cropping (defaults to 2176px)", metavar="int")

    outgroup = parser.add_argument_group("output control")
    outgroup.add_argument("-n", "--no_output", action="store_false", help="do not store any interim images (quicker)",
                          dest="save_files")
    outgroup.add_argument("--no_csv", action="store_false", help="do not store a csv of batch results", dest="save_csv")

    auxgroup = parser.add_argument_group("auxiliary commands")
    auxgroup.add_argument("--citation", action="store_true", help="print citations and exit")
    auxgroup.add_argument("--batchtest", nargs=4, help="Perform multicore batch test", metavar=("mincores", "maxcores", "repeats", "burnin?"))

    arglist = parser.parse_args()
    arglist = vars(arglist)
    if arglist["batchtest"]:
        try:
            burnin = bool(strtobool(arglist["batchtest"].pop(3)))
            batchnumargs = [int(x) for x in arglist["batchtest"]]
            logger.info("Test parameters\nCores={}-{}\nRepeats={}\nBurnin={}".format(*batchnumargs, burnin))
        except ValueError:
            logger.exception("Unable to convert numerical arguments %s to integers.", str(arglist["batchtest"]))
            sys.exit(1)
        testbatch(arglist, *batchnumargs,  burnin=burnin)
    else:
        main(arglist)
    sys.exit(0)
