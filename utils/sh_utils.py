#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5


"""
MIT License

Copyright (c) 2018 Andrew Chalmers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Spherical harmonics for radiance maps using numpy

Assumes:
Equirectangular format
theta: [0 to pi], from top to bottom row of pixels
phi: [0 to 2*Pi], from left to right column of pixels

"""
import os, sys
import numpy as np
import imageio as im
import cv2  # resize images with float support
from scipy import ndimage  # gaussian blur
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Spherical harmonics functions
def P(l, m, x):
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            fact += 2.0

    if l == m:
        return pmm * np.ones(x.shape)

    pmmp1 = x * (2.0 * m + 1.0) * pmm

    if l == m + 1:
        return pmmp1

    pll = np.zeros(x.shape)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def divfact(a, b):
    # PBRT style
    if b == 0:
        return 1.0
    fa = a
    fb = abs(b)
    v = 1.0

    x = fa - fb + 1.0
    while x <= fa + fb:
        v *= x
        x += 1.0

    return 1.0 / v


def factorial(x):
    if x == 0:
        return 1.0
    return x * factorial(x - 1)


def K(l, m):
    # return np.sqrt((2.0 * l + 1.0) * 0.07957747154594766788 * divfact(l, m))
    return np.sqrt(((2 * l + 1) * factorial(l - m)) / (4 * np.pi * factorial(l + m)))


def Kfast(l, m):
    cAM = abs(m)
    uVal = 1.0
    k = l + cAM
    while k > (l - cAM):
        uVal *= k
        k -= 1
    return np.sqrt((2.0 * l + 1.0) / (4 * np.pi * uVal))


def SH(l, m, theta, phi):
    sqrt2 = np.sqrt(2.0)
    if m == 0:
        if np.isscalar(phi):
            return K(l, m) * P(l, m, np.cos(theta))
        else:
            return K(l, m) * P(l, m, np.cos(theta)) * np.ones(phi.shape)
    elif m > 0:
        return sqrt2 * K(l, m) * np.cos(m * phi) * P(l, m, np.cos(theta))
    else:
        return sqrt2 * K(l, -m) * np.sin(-m * phi) * P(l, -m, np.cos(theta))


def SH_Inverse_Coefficient_Matrix(N, lmax):
    from scipy.special import lpmv
    theta = np.arccos(N[:, 1])
    phi = np.arctan(N[:, 0] / N[:, 2])
    phi[phi < 0] += 2 * np.pi

    coeffsMatrix = np.zeros((N.shape[0], shTerms(lmax)))
    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            if m > 0:
                coeffsMatrix[:, index] = np.cos(m * phi[:]) * lpmv(m, l, np.cos(theta[:]))
            elif m == 0:
                coeffsMatrix[:, index] = lpmv(m, l, np.cos(theta[:])) * np.ones(phi.shape)[:]
            elif m < 0:
                coeffsMatrix[:, index] = np.sin(-m * phi[:]) * lpmv(-m, l, np.cos(theta[:]))

    return coeffsMatrix


def shEvaluate(theta, phi, lmax):
    if np.isscalar(theta):
        coeffsMatrix = np.zeros((1, 1, shTerms(lmax)))
    else:
        coeffsMatrix = np.zeros((theta.shape[0], phi.shape[0], shTerms(lmax)))

    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            coeffsMatrix[:, :, index] = SH(l, m, theta, phi)
    return coeffsMatrix


def getCoefficientsMatrix(xres, lmax=2):
    yres = int(xres / 2)
    # setup fast vectorisation
    x = np.arange(0, xres)
    y = np.arange(0, yres).reshape(yres, 1)

    # Setup polar coordinates
    latLon = xy2ll(x, y, xres, yres)

    # Compute spherical harmonics. Apply thetaOffset due to EXR spherical coordiantes
    Ylm = shEvaluate(latLon[0], latLon[1], lmax)
    return Ylm


def getCoefficientsFromFile(ibl_filename, lmax=2, resizeWidth=None, filterAmount=None):
    ibl = im.imread(os.path.join(os.path.dirname(__file__), ibl_filename))
    return getCoefficientsFromImage(
        ibl, lmax=lmax, resizeWidth=resizeWidth, filterAmount=filterAmount
    )


def getCoefficientsFromImage(ibl, lmax=2, resizeWidth=None, filterAmount=None):
    # Resize if necessary (I recommend it for large images)
    if resizeWidth is not None:
        # ibl = cv2.resize(ibl, dsize=(resizeWidth,int(resizeWidth/2)), interpolation=cv2.INTER_CUBIC)
        ibl = resizeImage(ibl, resizeWidth, int(resizeWidth / 2), cv2.INTER_CUBIC)
    elif ibl.shape[1] > 1000:
        # print("Input resolution is large, reducing for efficiency")
        # ibl = cv2.resize(ibl, dsize=(1000,500), interpolation=cv2.INTER_CUBIC)
        ibl = resizeImage(ibl, 1000, 500, cv2.INTER_CUBIC)
    xres = ibl.shape[1]
    yres = ibl.shape[0]

    # Pre-filtering, windowing
    if filterAmount is not None:
        ibl = blurIBL(ibl, amount=filterAmount)

    # Compute sh coefficients
    sh_basis_matrix = getCoefficientsMatrix(xres, lmax)

    # Sampling weights
    solidAngles = getSolidAngleMap(xres)

    # Project IBL into SH basis
    nCoeffs = shTerms(lmax)
    iblCoeffs = np.zeros((nCoeffs, 3))
    for i in range(0, shTerms(lmax)):
        iblCoeffs[i, 0] = np.sum(ibl[:, :, 0] * sh_basis_matrix[:, :, i] * solidAngles)
        iblCoeffs[i, 1] = np.sum(ibl[:, :, 1] * sh_basis_matrix[:, :, i] * solidAngles)
        iblCoeffs[i, 2] = np.sum(ibl[:, :, 2] * sh_basis_matrix[:, :, i] * solidAngles)

    return iblCoeffs


def findWindowingFactor(coeffs, maxLaplacian=10.0):
    # http://www.ppsloan.org/publications/StupidSH36.pdf
    # Based on probulator implementation, empirically chosen maxLaplacian
    lmax = sh_lmax_from_terms(coeffs.shape[0])
    tableL = np.zeros((lmax + 1))
    tableB = np.zeros((lmax + 1))

    def sqr(x):
        return x * x

    def cube(x):
        return x * x * x

    for l in range(1, lmax + 1):
        tableL[l] = float(sqr(l) * sqr(l + 1))
        B = 0.0
        for m in range(-1, l + 1):
            B += np.mean(coeffs[shIndex(l, m), :])
        tableB[l] = B

    squaredLaplacian = 0.0
    for l in range(1, lmax + 1):
        squaredLaplacian += tableL[l] * tableB[l]

    targetSquaredLaplacian = maxLaplacian * maxLaplacian
    if squaredLaplacian <= targetSquaredLaplacian:
        return 0.0

    windowingFactor = 0.0
    iterationLimit = 10000000
    for i in range(0, iterationLimit):
        f = 0.0
        fd = 0.0
        for l in range(1, lmax + 1):
            f += tableL[l] * tableB[l] / sqr(1.0 + windowingFactor * tableL[l])
            fd += (2.0 * sqr(tableL[l]) * tableB[l]) / cube(
                1.0 + windowingFactor * tableL[l]
            )

        f = targetSquaredLaplacian - f

        delta = -f / fd
        windowingFactor += delta
        if abs(delta) < 0.0000001:
            break
    return windowingFactor


def applyWindowing(coeffs, windowingFactor=None, verbose=False):
    # http://www.ppsloan.org/publications/StupidSH36.pdf
    lmax = sh_lmax_from_terms(coeffs.shape[0])
    if windowingFactor is None:
        windowingFactor = findWindowingFactor(coeffs)
    if windowingFactor <= 0:
        if verbose:
            print("No windowing applied")
        return coeffs
    if verbose:
        print("Using windowingFactor: %s" % (windowingFactor))
    for l in range(0, lmax + 1):
        s = 1.0 / (1.0 + windowingFactor * l * l * (l + 1.0) * (l + 1.0))
        for m in range(-l, l + 1):
            coeffs[shIndex(l, m), :] *= s
    return coeffs


# Misc functions
def resizeImage(img, width, height, interpolation=cv2.INTER_CUBIC):
    if img.shape[1] < width:  # up res
        if interpolation == "max_pooling":
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            return cv2.resize(img, (width, height), interpolation=interpolation)
    if interpolation == "max_pooling":  # down res, max pooling
        try:
            import skimage.measure

            scaleFactor = int(float(img.shape[1]) / width)
            factoredWidth = width * scaleFactor
            img = cv2.resize(
                img,
                (factoredWidth, int(factoredWidth / 2)),
                interpolation=cv2.INTER_CUBIC,
            )
            blockSize = scaleFactor
            r = skimage.measure.block_reduce(
                img[:, :, 0], (blockSize, blockSize), np.max
            )
            g = skimage.measure.block_reduce(
                img[:, :, 1], (blockSize, blockSize), np.max
            )
            b = skimage.measure.block_reduce(
                img[:, :, 2], (blockSize, blockSize), np.max
            )
            img = np.dstack((np.dstack((r, g)), b)).astype(np.float32)
            return img
        except:
            print("Failed to do max_pooling, using default")
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    else:  # down res, using interpolation
        return cv2.resize(img, (width, height), interpolation=interpolation)


def grey2colour(greyImg):
    return (np.repeat(greyImg[:, :][:, :, np.newaxis], 3, axis=2)).astype(np.float32)


def colour2grey(colImg):
    return ((colImg[:, :, 0] + colImg[:, :, 1] + colImg[:, :, 2]) / 3).astype(
        np.float32
    )


def poleScale(y, width, relative=True):
    height = int(width / 2)
    piHalf = np.pi / 2
    pi4 = np.pi * 4
    pi2OverWidth = (np.pi * 2) / width
    piOverHeight = np.pi / height
    theta = (1.0 - ((y + 0.5) / height)) * np.pi
    scaleFactor = (
            (1.0 / pi4)
            * pi2OverWidth
            * (np.cos(theta - (piOverHeight / 2.0)) - np.cos(theta + (piOverHeight / 2.0)))
    )
    if relative:
        scaleFactor /= (
                (1.0 / pi4)
                * pi2OverWidth
                * (
                        np.cos(piHalf - (piOverHeight / 2.0))
                        - np.cos(piHalf + (piOverHeight / 2.0))
                )
        )
    return scaleFactor


def getSolidAngle(y, width, is3D=False):
    height = int(width / 2)
    pi2OverWidth = (np.pi * 2) / width
    piOverHeight = np.pi / height
    theta = (1.0 - ((y + 0.5) / height)) * np.pi
    return pi2OverWidth * (
            np.cos(theta - (piOverHeight / 2.0)) - np.cos(theta + (piOverHeight / 2.0))
    )


def getSolidAngleMap(width):
    height = int(width / 2)
    return np.repeat(
        getSolidAngle(np.arange(0, height), width)[:, np.newaxis], width, axis=1
    )


def getDiffuseMap(ibl_name, width=600, widthLowRes=32, outputWidth=None):
    if outputWidth is None:
        outputWidth = width
    height = int(width / 2)
    heightLowRes = int(widthLowRes / 2)

    img = im.imread(ibl_name, "EXR-FI")[:, :, 0:3]
    img = resizeImage(img, width, height)

    uv_x = np.arange(float(width)) / width
    uv_x = np.tile(uv_x, (height, 1))

    uv_y = np.arange(float(height)) / height
    uv_y = 1 - np.tile(uv_y, (width, 1)).transpose()

    phi = np.pi * (uv_y - 0.5)
    theta = 2 * np.pi * (1 - uv_x)

    cos_phi = np.cos(phi)
    d_x = cos_phi * np.sin(theta)
    d_y = np.sin(phi)
    d_z = cos_phi * np.cos(theta)

    solidAngles = getSolidAngleMap(width)

    print("Convolving ", (widthLowRes, heightLowRes))
    outputDiffuseMap = np.zeros((heightLowRes, widthLowRes, 3))

    def compute(x_i, y_i):
        x_i_s = int((float(x_i) / widthLowRes) * width)
        y_i_s = int((float(y_i) / heightLowRes) * height)
        dot = np.maximum(
            0,
            d_x[y_i_s, x_i_s] * d_x + d_y[y_i_s, x_i_s] * d_y + d_z[y_i_s, x_i_s] * d_z,
        )
        for c_i in range(0, 3):
            outputDiffuseMap[y_i, x_i, c_i] = (
                    np.sum(dot * img[:, :, c_i] * solidAngles) / np.pi
            )

    start = time.time()
    for x_i in range(0, outputDiffuseMap.shape[1]):
        # print(float(x_i)/outputDiffuseMap.shape[1])
        for y_i in range(0, outputDiffuseMap.shape[0]):
            compute(x_i, y_i)
    end = time.time()
    print("Elapsed time: %.4f seconds" % (end - start))

    if widthLowRes < outputWidth:
        outputDiffuseMap = resizeImage(
            outputDiffuseMap, outputWidth, int(outputWidth / 2), cv2.INTER_LANCZOS4
        )

    return outputDiffuseMap.astype(np.float32)


# Spherical harmonics reconstruction
def getDiffuseCoefficients(lmax):
    # From "An Efficient Representation for Irradiance Environment Maps" (2001), Ramamoorthi & Hanrahan
    diffuseCoeffs = [np.pi, (2 * np.pi) / 3]
    for l in range(2, lmax + 1):
        if l % 2 == 0:
            a = (-1.0) ** ((l / 2.0) - 1.0)
            b = (l + 2.0) * (l - 1.0)
            c = float(np.math.factorial(l)) / (2 ** l * np.math.factorial(l / 2) ** 2)
            # s = ((2*l+1)/(4*np.pi))**0.5
            diffuseCoeffs.append(2 * np.pi * (a / b) * c)
        else:
            diffuseCoeffs.append(0)
    return np.asarray(diffuseCoeffs) / np.pi


def shReconstructSignal(coeffs, sh_basis_matrix=None, width=600):
    if sh_basis_matrix is None:
        lmax = sh_lmax_from_terms(coeffs.shape[0])
        sh_basis_matrix = getCoefficientsMatrix(width, lmax)
    return np.dot(sh_basis_matrix, coeffs).astype(np.float32)


def shRender(iblCoeffs, width=600):
    lmax = sh_lmax_from_terms(iblCoeffs.shape[0])
    diffuseCoeffs = getDiffuseCoefficients(lmax)
    sh_basis_matrix = getCoefficientsMatrix(width, lmax)
    renderedImage = np.zeros((int(width / 2), width, 3))
    for idx in range(0, iblCoeffs.shape[0]):
        l = l_from_idx(idx)
        coeff_rgb = diffuseCoeffs[l] * iblCoeffs[idx, :]
        renderedImage[:, :, 0] += sh_basis_matrix[:, :, idx] * coeff_rgb[0]
        renderedImage[:, :, 1] += sh_basis_matrix[:, :, idx] * coeff_rgb[1]
        renderedImage[:, :, 2] += sh_basis_matrix[:, :, idx] * coeff_rgb[2]
    return renderedImage


def getNormalMapAxesDuplicateRGB(normalMap):
    # Make normal for each axis, but in 3D so we can multiply against RGB
    N3Dx = np.repeat(normalMap[:, :, 0][:, :, np.newaxis], 3, axis=2)
    N3Dy = np.repeat(normalMap[:, :, 1][:, :, np.newaxis], 3, axis=2)
    N3Dz = np.repeat(normalMap[:, :, 2][:, :, np.newaxis], 3, axis=2)
    return N3Dx, N3Dy, N3Dz


def shRenderL2(iblCoeffs, normalMap):
    # From "An Efficient Representation for Irradiance Environment Maps" (2001), Ramamoorthi & Hanrahan
    C1 = 0.429043
    C2 = 0.511664
    C3 = 0.743125
    C4 = 0.886227
    C5 = 0.247708
    N3Dx, N3Dy, N3Dz = getNormalMapAxesDuplicateRGB(normalMap)
    return (
            C4 * iblCoeffs[0, :]
            + 2.0 * C2 * iblCoeffs[3, :] * N3Dx
            + 2.0 * C2 * iblCoeffs[1, :] * N3Dy
            + 2.0 * C2 * iblCoeffs[2, :] * N3Dz
            + C1 * iblCoeffs[8, :] * (N3Dx * N3Dx - N3Dy * N3Dy)
            + C3 * iblCoeffs[6, :] * N3Dz * N3Dz
            - C5 * iblCoeffs[6]
            + 2.0 * C1 * iblCoeffs[4, :] * N3Dx * N3Dy
            + 2.0 * C1 * iblCoeffs[7, :] * N3Dx * N3Dz
            + 2.0 * C1 * iblCoeffs[5, :] * N3Dy * N3Dz
    ) / np.pi


def getNormalMap(width):
    height = int(width / 2)
    x = np.arange(0, width)
    y = np.arange(0, height).reshape(height, 1)
    latLon = xy2ll(x, y, width, height)
    return spherical2Cartesian2(latLon[0], latLon[1])


def shReconstructDiffuseMap(iblCoeffs, width=600):
    # Rendering
    if iblCoeffs.shape[0] == 9:  # L2
        # setup fast vectorisation
        xyz = getNormalMap(width)
        renderedImage = shRenderL2(iblCoeffs, xyz)
    else:  # !L2
        renderedImage = shRender(iblCoeffs, width)

    return renderedImage.astype(np.float32)


def shReconstructDiffuseNormalMap(iblCoeffs, normal_map):
    # Rendering
    renderedImage = shRenderL2(iblCoeffs, normal_map)
    return renderedImage.astype(np.float32)


def writeReconstruction(c, lmax, fn="", width=600, outputDir="./output/"):
    im.imwrite(
        outputDir + "_sh_light_l" + str(lmax) + fn + ".exr",
        shReconstructSignal(c, width=width),
    )
    im.imwrite(
        outputDir + "_sh_render_l" + str(lmax) + fn + ".exr",
        shReconstructDiffuseMap(c, width=width),
    )


# Utility functions for SPH
def shPrint(coeffs, precision=3):
    nCoeffs = coeffs.shape[0]
    lmax = sh_lmax_from_terms(coeffs.shape[0])
    currentBand = -1
    for idx in range(0, nCoeffs):
        band = l_from_idx(idx)
        if currentBand != band:
            currentBand = band
            print("L" + str(currentBand) + ":")
        print(np.around(coeffs[idx, :], precision))
    print("")


def shTermsWithinBand(l):
    return (l * 2) + 1


def shTerms(lmax):
    return (lmax + 1) * (lmax + 1)


def sh_lmax_from_terms(terms):
    return int(np.sqrt(terms) - 1)


def shIndex(l, m):
    return l * l + l + m


def l_from_idx(idx):
    return int(np.sqrt(idx))


def paintNegatives(img):
    indices = [img[:, :, 0] < 0 or img[:, :, 1] < 0 or img[:, :, 2] < 0]
    img[indices[0], 0] = (
            abs((img[indices[0], 0] + img[indices[0], 1] + img[indices[0], 2]) / 3) * 10
    )
    img[indices[0], 1] = 0
    img[indices[0], 2] = 0


def blurIBL(ibl, amount=5):
    x = ibl.copy()
    x[:, :, 0] = ndimage.gaussian_filter(ibl[:, :, 0], sigma=amount)
    x[:, :, 1] = ndimage.gaussian_filter(ibl[:, :, 1], sigma=amount)
    x[:, :, 2] = ndimage.gaussian_filter(ibl[:, :, 2], sigma=amount)
    return x


def spherical2Cartesian2(theta, phi):
    phi = phi + np.pi
    x = np.sin(theta) * np.cos(phi)
    y = np.cos(theta)
    z = np.sin(theta) * np.sin(phi)
    if not np.isscalar(x):
        y = np.repeat(y, x.shape[1], axis=1)
    return np.moveaxis(np.asarray([x, z, y]), 0, 2)


def spherical2Cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    if not np.isscalar(x):
        z = np.repeat(z, x.shape[1], axis=1)
    return np.moveaxis(np.asarray([x, z, y]), 0, 2)


def xy2ll(x, y, width, height):
    def yLocToLat(yLoc, height):
        return yLoc / (float(height) / np.pi)

    def xLocToLon(xLoc, width):
        return xLoc / (float(width) / (np.pi * 2))

    return np.asarray([yLocToLat(y, height), xLocToLon(x, width)], dtype=object)


def getCartesianMap(width):
    height = int(width / 2)
    image = np.zeros((height, width))
    x = np.arange(0, width)
    y = np.arange(0, height).reshape(height, 1)
    latLon = xy2ll(x, y, width, height)
    return spherical2Cartesian(latLon[0], latLon[1])


# Example functions
def cosine_lobe_example(dir, width):
    # https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc
    xyz = getCartesianMap(width)
    return grey2colour(np.clip(np.sum(dir * xyz, axis=2), 0.0, 1.0))
    # return grey2colour(1.0 * (np.exp(-(np.arccos(np.sum(dir*xyz, axis=2))/(0.5**2)))))


def robin_green_example(latLon, width, height):
    # "The Gritty Details" by Robin Green
    dir1 = np.asarray([0, 1, 0])
    theta = np.repeat(latLon[0][:, np.newaxis], width, axis=1).reshape((height, width))
    phi = np.repeat(latLon[1][np.newaxis, :], height, axis=0).reshape((height, width))
    return grey2colour(
        np.maximum(0.0, 5 * np.cos(theta) - 4)
        + np.maximum(0.0, -4 * np.sin(theta - np.pi) * np.cos(phi - 2.5) - 3)
    )


# Visualisations
def sh_visualise(lmax=2, sh_basis_matrix=None, showIt=False, outputDir="./output/"):
    cdict = {
        "red": ((0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "green": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)),
        "blue": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    }

    if sh_basis_matrix is None:
        sh_basis_matrix = getCoefficientsMatrix(600, lmax)

    lmax = sh_lmax_from_terms(sh_basis_matrix.shape[2])

    rows = lmax + 1
    cols = shTermsWithinBand(lmax)
    imgIndex = 0

    if lmax == 0:
        plt.imshow(
            sh_basis_matrix[:, :, 0],
            cmap=LinearSegmentedColormap("RedGreen", cdict),
            vmin=-1,
            vmax=1,
        )
        plt.axis("off")
        plt.savefig(outputDir + "_fig_sh_l" + str(lmax) + ".jpg")
        if showIt:
            plt.show()
        return

    _, axs = plt.subplots(
        nrows=rows,
        ncols=cols,
        gridspec_kw={"wspace": 0.1, "hspace": -0.4},
        squeeze=True,
        figsize=(16, 8),
    )
    for c in range(0, cols):
        for r in range(0, rows):
            axs[r, c].axis("off")

    for l in range(0, lmax + 1):
        nInBand = shTermsWithinBand(l)
        colOffset = int(cols / 2) - int(nInBand / 2)
        rowOffset = (l * cols) + 1
        index = rowOffset + colOffset
        for i in range(0, nInBand):
            axs[l, i + colOffset].axis("off")
            axs[l, i + colOffset].imshow(
                sh_basis_matrix[:, :, imgIndex],
                cmap=LinearSegmentedColormap("RedGreen", cdict),
                vmin=-1,
                vmax=1,
            )
            imgIndex += 1

    plt.savefig(outputDir + "_fig_sh_l" + str(lmax) + ".jpg")
    if showIt:
        plt.show()



def factorialTorch(x):
    if x == 0:
        return 1.0
    return x * factorialTorch(x - 1)


def PTorch(l, m, x, device):
    pmm = 1.0
    if m > 0:
        somx2 = torch.sqrt((1.0 - x) * (1.0 + x)).to(device)
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= (-fact) * somx2
            fact += 2.0

    if l == m:
        return pmm * torch.ones(x.shape).to(device)

    pmmp1 = x * (2.0 * m + 1.0) * pmm

    if l == m + 1:
        return pmmp1

    pll = torch.zeros(x.shape).to(device)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll


def shTermsTorch(lmax):
    return (lmax + 1) * (lmax + 1)


def KTorch(l, m, device):
    return torch.sqrt(
        torch.tensor(
            ((2 * l + 1) * factorialTorch(l - m))
            / (4 * torch.pi * factorialTorch(l + m))
        )
    ).to(device)


def shIndexTorch(l, m):
    return l * l + l + m


def SHTorch(l, m, theta, phi, device):
    sqrt2 = np.sqrt(2.0)
    if m == 0:
        return (
                KTorch(l, m, device)
                * PTorch(l, m, torch.cos(theta), device)
                * torch.ones(phi.shape).to(device)
        )
    elif m > 0:
        return (
                sqrt2
                * KTorch(l, m, device)
                * torch.cos(m * phi)
                * PTorch(l, m, torch.cos(theta), device)
        )
    else:
        return (
                sqrt2
                * KTorch(l, -m, device)
                * torch.sin(-m * phi)
                * PTorch(l, -m, torch.cos(theta), device)
        )


def shEvaluateTorch(theta, phi, lmax, device):
    coeffsMatrix = torch.zeros((theta.shape[0], phi.shape[0], shTermsTorch(lmax))).to(
        device
    )

    for l in range(0, lmax + 1):
        for m in range(-l, l + 1):
            index = shIndexTorch(l, m)
            coeffsMatrix[:, :, index] = SHTorch(l, m, theta, phi, device)
    return coeffsMatrix


def xy2llTorch(x, y, width, height):
    def yLocToLat(yLoc, height):
        return yLoc / (float(height) / torch.pi)

    def xLocToLon(xLoc, width):
        return xLoc / (float(width) / (torch.pi * 2))

    return yLocToLat(y, height), xLocToLon(x, width)


def getCoefficientsMatrixTorch(xres, lmax, device):
    yres = int(xres / 2)
    # setup fast vectorisation
    x = torch.arange(0, xres).to(device)
    y = torch.arange(0, yres).reshape(yres, 1).to(device)

    # Setup polar coordinates
    lat, lon = xy2llTorch(x, y, xres, yres)

    # Compute spherical harmonics. Apply thetaOffset due to EXR spherical coordiantes
    Ylm = shEvaluateTorch(lat, lon, lmax, device)
    return Ylm


def sh_lmax_from_termsTorch(terms):
    return int(torch.sqrt(terms) - 1)


def shReconstructSignalTorch(coeffs, width, device):
    lmax = sh_lmax_from_termsTorch(torch.tensor(coeffs.shape[0]).to(device))
    sh_basis_matrix = getCoefficientsMatrixTorch(width, lmax, device)
    return torch.einsum("ijk,kl->ijl", sh_basis_matrix, coeffs)  # (H, W, 3)


def calc_num_sh_coeffs(order):
    coeffs = 0
    for i in range(order + 1):
        coeffs += 2 * i + 1
    return coeffs


def get_sh_order(ndims):
    order = 0
    while calc_num_sh_coeffs(order) < ndims:
        order += 1
    return order


def get_spherical_harmonic_representation(img, nBands):
    # img: (H, W, 3), nBands: int
    iblCoeffs = getCoefficientsFromImage(img, nBands)
    sh_radiance_map = shReconstructSignal(
        iblCoeffs, width=img.shape[1]
    )
    sh_radiance_map = torch.from_numpy(sh_radiance_map)
    return sh_radiance_map

