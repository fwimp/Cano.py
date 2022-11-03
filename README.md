# Cano.py Documentation

Welcome to Cano.py (hereafter referred to as Cano), a tool written in Python 3 and scikit-image to automate the projection and analysis of cylindrical canopy images.

Cano will crop an image, reproject it in hemispherical space, and calculate LAI from the canopy cover.

The following guide is intended to provide a relatively comprehensive overview of the command line interface to Cano alongside providing examples of general use pipelines.

---

## Environment setup
Coming soon...

---

## Using the CLI
Cano is a purely command line program. Whilst this can be intimidating, it is intended to give the maximum flexibility for both users and developers. 

Cano is run from the command line using the following template

`Cano.py  <IMAGE> <OPTIONS>`

`<IMAGE>` refers to the panoramic image/s that you wish to process (more on that later)

`<OPTIONS>` is any option you would like to provide to control the way that Cano runs.


For example if you want to run Cano
- In multicore mode on 8 cores
- In debug mode
- Picking up some previously standardised images from the directory `../results/polar`
- With an LAI threshold of 0.7 
- Without storing any of the threshold images

you could run the following:

`Cano.py ./results/polar --multicore 8 --debug --pickup --threshold 0.7 --no_output`

Alternatively we can use shortened versions of a lot of these options:

`Cano.py ./results/polar -c 8 -d -p -t 0.7 -n`

Futhermore we can combine together options that do not need a value into one argument like so:

`Cano.py ./results/polar -dpn -c 8 -t 0.7`

Finally some options (such as `--no_csv`) do not have a shorthand, in which case we can still use the long form in conjunction with other short forms:

`Cano.py ./results/polar -dpn --multicore 8 --threshold 0.7 --no_csv`

## CLI Documentation

---

### Help
`-h / --help`

Prints the help file for Cano (as below)

```
usage: Cano.py [-h] [-o [d]] [-m | -p] [-c [int]] [-d] [-t [flt]] [-s [int]] [-n] [--no_csv]
               [--citation]
               image

Transform, threshold, and calculate LAI for panoramic canopy photos.

positional arguments:
  image                 an image to analyse (or folder containing said images)

optional arguments:
  -h, --help            show this help message and exit
  -o [d], --outdir [d]  output directory
  -m, --midpoint        output polar image for standardisation (cannot be combined with -p)
  -p, --pickup          pick up from standardised polar images for thresholding and LAI calculation
                        (cannot be combined with -m)
  -c [int], --multicore [int]
                        enable multicore processing
  -d, --debug           enable debugging information

processing parameters:
  -t [flt], --threshold [flt]
                        threshold proportion for LAI calculation (defaults to 0.82)
  -s [int], --slice [int]
                        slice point for image cropping (defaults to 2176px)

output control:
  -n, --no_output       do not store any interim images (quicker)
  --no_csv              do not store a csv of batch results

auxiliary commands:
  --citation            print citations and exit
```

### Image file
`image`

The image parameter is the only required argument to Cano. In normal operation this is expected to be the file path to the image that you wish to process.

If you would like to process a folder of images, provide that as an argument instead, and Cano.py will intelligently process in batch mode instead.

### Output directory
`-o <DIR> / --outdir <DIR>`

Output directories are inferred based upon the assumption that your directory is set out approximately as follows:

```
analysis_folder
├── Cano.py
├── data
│   ├── file1.jpg
│   ├── file2.jpg
│   └── ...
└── results 
```

If this is not what you want, you can provide an output directory using `-o` which Cano will use as its results folder, like so:

`Cano.py image.jpg --outdir ./path/to/results`

### Midpoint / Pickup processing
`-m / --midpoint` or `-p / --pickup`

**These options are mutually exclusive.**

By default Cano will perform a full beginning-to-end processing of a panoramic image or set of images.

If you wish to perform image standardisation on the polar image prior to thresholding and retrieving LAI metrics, you should first run Cano using the `-m` option:

`Cano.py ./data/image.jpg -m`

This will perform the initial cropping and projection steps, before saving the results to the `results/polar` folder.
You can then standardise these pictures at will using any appropriate method.

Once you have completed the standardisation and saved the _images_ into an appropriate location, you should then run Cano again with the `-p` flag:

`Cano.py ./results/polar/image_standardised.jpg -p`

This will pick up these standardised files and use them as the base for the thresholding and LAI calculation.

### Threshold
`-t [flt] / --threshold [flt]`

The threshold option determines the threshold for LAI calculation. By default this is set to 0.82.

LAI is sensitive to the threshold value, however, so if you wish to change this, use the `-t` option.

### Slice
`-s [int] / --slice [int]`

The slice option controls the distance from the top of the image in pixels to crop to. By default this is set to 2176px.

If you have a different setup to the default then this might need to be tuned accordingly using the `-s` option.

### No Intermediate Image Output
`-n / --no_output`

By default, Cano saves the intermediate polar projection images to the folder `results/polar/` and the threshold images to `results/thresh/`.

If you do not need these images you can save a bit of execution time and storage space using the `-n` option.

### No CSV
`--no_csv`

Usually, Cano outputs the results of a batch process (i.e. using more than 1 image) as a csv file in the results folder.

If you do not wish this for whatever reason, you can disable the csv output using `--no_csv`

### Multicore processing
`-c <CORES> / --multicore <CORES>`

Cano includes support for multi-core processing on appropriate cpus.

By default if you do not supply a number of cores, Cano will use one less than the maximum number of cores in your system (to allow you to still navigate around a bit if needed.)

If you wish to use a specific number of cores, you can provide it as a value after the `-c`.

*Cano will not try to use more cores than your system has (less 1) even if you ask it nicely.*

This can speed up large batch jobs by an order of magnitude, however this does come at the cost of high memory usage.
Processing images of a size 8704 x 4352px in single-core mode takes approximately 2.4GB of RAM. This memory usage multiplies linearly with the number of cores used, as seen in the table below:

For 11* images at 8704x4352px (with no saving of intermediate images), 5 repeats at each core number:

| Cores |  Time  |  SD   | Predicted Memory | Time/Image |
|------:|:------:|:-----:|:----------------:|:----------:|
|     1 | 83.98s | 0.64s |      2.4GB       |   7.63s    |
|     2 | 49.49s | 0.21s |      4.8GB       |   4.50s    |
|     3 | 34.74s | 0.09s |      7.2GB       |   3.16s    |
|     4 | 27.85s | 0.21s |      9.6GB       |   2.53s    |
|     5 | 28.38s | 0.14s |       12GB       |   2.58s    |
|     6 | 21.83s | 0.30s |      14.4GB      |   1.98s    |
|     7 | 23.30s | 0.26s |      16.8GB      |   2.12s    |
|     8 | 23.49s | 0.16s |      19.2GB      |   2.14s    |
|     9 | 24.41s | 0.16s |      21.6GB      |   2.22s    |
|    10 | 25.27s | 0.23s |       24GB       |   2.30s    |
|   11* | 19.88s | 0.45s |      26.4GB      |   1.80s    |
|    12 | 19.58s | 0.27s |      28.8GB      |   1.78s    |
|    13 | 19.70s | 0.15s |      31.2GB      |   1.79s    |
|    14 | 19.48s | 0.17s |      33.6GB      |   1.77s    |
|    15 | 20.21s | 0.33s |       36GB       |   1.84s    |
*(On a 16 core, 32GB system (34% used at runtime))*

In general a good rule of thumb is that the number of cores you use should be either equivalent to:
- The number of images you are processing, *or*
- The largest (rounded up) whole-number divisor of the number of images within the number of cores you have, *or*
- The largest whole-number divisor of the number of images that you can support in terms of RAM

So if you are processing **40** images on a **16 core** cpu with **32GB of RAM**, you would probably want to use a core count of **10** (so each cpu processes 4 images).

If you were processing 151 images on a 16 core cpu, you might consider using 8 cores, or 14 if you are happy with the RAM requirement.

#### Notes:
*Usually you get significantly diminishing returns on increasing core count past around 4. If you get errors, system instability, or other problems when using high core counts, drop the number to something more reasonable.*

*Smaller images have a significantly lower RAM and processing requirement, so do consider that if you are having problems, but remember to change the `--slice` command appropriately.*

### Debug Mode
`-d / --debug`

Enabling debug mode prints a bit more information to the terminal when the program is running.

This is not usually necessary for day-to-day use.

### Citations
`--citation`

Cano.py has been built on the work of previous researchers.  In order to preserve their contributions a citation argument has been added that prints the relevant citations.

Additionally the citations are attached here for your convenience

```
Cano.py is a wrapper and CLI for a multicore adaptation of a previous
digital hemispherical photography (DHP) analysis pipeline.

This pipeline underlies the code of https://app.cano.fi/
originally by Jon Atherton (University of Helsinki)
https://www.doi.org/10.5281/zenodo.5171970
https://github.com/joathert/canofi-app

The LAI inference is based upon Hemiphot.R:
Hans ter Steege (2018)
Hemiphot.R: Free R scripts to analyse hemispherical photographs for canopy openness,
leaf area index and photosynthetic active radiation under forest canopies.
Unpublished report. Naturalis Biodiversity Center, Leiden, The Netherlands
https://github.com/Hans-ter-Steege/Hemiphot

CLI, multiprocessing, optimisation, and extra programming
by Francis Windram, Imperial College London
```


## Known Bugs

---
### Multicore runs are impossible to exit early
This is definitely a problem, but one with a non-trivial solution, [see here for more details.](https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool/2561809#2561809)

At present, sadly the best solution is to quit the terminal window and then kill all python processes still running if necessary.

In general, try to make sure that any big multicore jobs are properly formatted and parameterised. You could also run a small subset in single-core mode to check before switching to multicore for the main run.

**IF YOU POSSIBLY CAN, LET MULTICORE JOBS RUN ENTIRELY, EVEN IF THEY ARE WRONG!**

---

### Slight differences in LAI between midpoint/pickup runs and full runs
Some slight differences in LAI have been detected between runs that go all the way through vs. ones that are split up.

Right now it is unclear precisely what causes this, though it is likely to be an issue with compression of the colour space.  These errors seem pretty small, and in comparison to the gains in any standardisation steps they should not matter too much.