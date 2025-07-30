from PIL import Image, ImageDraw
import numpy as np
from skimage.metrics import structural_similarity
import cv2
import matplotlib.pyplot as plt
from math import log10, sqrt
from sewar.full_ref import uqi


class ImageHelper:

    def __init__(self, imagePath, polygonSize=3):
        """
        Initializes an instance of the class
        :param imagePath: the path of the file containing the reference image
        :param polygonSize: the number of vertices on the polygons used to recreate the image
        """
        self.refImage = Image.open(imagePath)
        self.polygonSize = polygonSize

        self.width, self.height = self.refImage.size
        self.numPixels = self.width * self.height
        self.refImageCv2 = self.toCv2(self.refImage)

    def polygonDataToImage(self, polygonData):
        """
        accepts polygon data and creates an image containing these polygons.
        :param polygonData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color and transparency of the corresponding polygon
        :return: the image containing the polygons (Pillow format)
        """

        # start with a new image:
        image = Image.new('RGB', (self.width, self.height))  # TODO
        draw = ImageDraw.Draw(image, 'RGBA')

        # divide the polygonData to chunks, each containing the data for a single polygon:
        chunkSize = self.polygonSize * 2 + 4  # (x,y) per vertex + (RGBA)
        polygons = self.list2Chunks(polygonData, chunkSize)

        # iterate over all polygons and draw each of them into the image:
        for poly in polygons:
            index = 0

            # extract the vertices of the current polygon:
            vertices = []
            for vertex in range(self.polygonSize):
                vertices.append(
                    (int(poly[index] * self.width), int(poly[index + 1] * self.height)))
                index += 2

            # extract the RGB and alpha values of the current polygon:
            red = int(poly[index] * 255)
            green = int(poly[index + 1] * 255)
            blue = int(poly[index + 2] * 255)
            alpha = int(poly[index + 3] * 255)

            # draw the polygon into the image:
            draw.polygon(vertices, (red, green, blue, alpha))

        # cleanup:
        del draw

        return image

    def getDifferenceFunc(self, method="MSE"):
        """
        accepts polygon data, creates an image containing these polygons, and calculates the difference
        between this image and the reference image using one of two methods.
        :param polygonData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color and transparency of the corresponding polygon
        :param method: base method of calculating the difference ("MSE" or "SSIM" or "PSNR").
        larger return value always means larger difference
        :return: the calculated difference between the image containg the polygons and the reference image
        """
        def _internal_mse(polygonData):
            image = self.polygonDataToImage(polygonData)
            return self.getMse(image)

        def _internal_ssim(polygonData):
            image = self.polygonDataToImage(polygonData)
            return 1.0 - self.getSsim(image)

        def _internal_psnr(polygonData):
            image = self.polygonDataToImage(polygonData)
            return 1.0 - (self.getPSNR(image)/100)

        def _internal_loss(polygonData):
            image = self.polygonDataToImage(polygonData)
            return self.getLoss(image)

        def _internal_cp(polygonData):
            image = self.polygonDataToImage(polygonData)
            return 1/(self.getCP(image)+1)

        if method == "MSE":
            return _internal_mse
        elif method == "SSIM":
            return _internal_ssim
        elif method == "PSNR":
            return _internal_psnr
        elif method == "LOSS":
            return _internal_loss
        elif method == "CP":
            return _internal_cp
        else:
            raise Exception("Method not supported")

    def symiliarityMethods():
        return ["MSE", "SSIM", "PSNR", "LOSS", "CP", "UQI"]

    def getDifference(self, polygonData, method="MSE"):
        """
        accepts polygon data, creates an image containing these polygons, and calculates the difference
        between this image and the reference image using one of two methods.
        :param polygonData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color and transparency of the corresponding polygon
        :param method: base method of calculating the difference ("MSE" or "SSIM" or "PSNR").
        larger return value always means larger difference
        :return: the calculated difference between the image containg the polygons and the reference image
        """

        # create the image containing the polygons:
        image = self.polygonDataToImage(polygonData)

        if method == "MSE":
            return self.getMse(image)
        elif method == "SSIM":
            return 1.0 - self.getSsim(image)
        elif method == "PSNR":
            return 1.0 - (self.getPSNR(image)/100)
        elif method == "LOSS":
            return self.getLoss(image)
        elif method == "CP":
            return 1/(self.getCP(image)+1)
        else:
            raise Exception("Method not supported")

    def plotImages(self, image, header=None):
        """
        creates a 'side-by-side' plot of the given image next to the reference image
        :param image: image to be drawn next to reference image (Pillow format)
        :param header: text used as a header for the plot
        """

        fig = plt.figure("Image Comparison:", clear=True)
        if header:
            fig.suptitle(header)

        # plot the reference image on the left:
        ax = fig.add_subplot(1, 2, 1)
        fig.gca().imshow(self.refImage)
        self.ticksOff(plt)

        # plot the given image on the right:
        fig.add_subplot(1, 2, 2)
        fig.gca().imshow(image)
        self.ticksOff(plt)

        return fig

    def saveImage(self, polygonData, imageFilePath, header=None):
        """
        accepts polygon data, creates an image containing these polygons,
        creates a 'side-by-side' plot of this image next to the reference image,
        and saves the plot to a file
        :param polygonData: a list of polygon parameters. Each item in the list
        represents the vertices locations, color and transparency of the corresponding polygon
        :param imageFilePath: path of file to be used to save the plot to
        :param header: text used as a header for the plot
        """
        # create an image from th epolygon data:
        image = self.polygonDataToImage(polygonData)

        # plot the image side-by-side with the reference image:
        fig = self.plotImages(image, header)
        fig.savefig(imageFilePath)
        plt.close(fig)

    # utility methods:
    def toCv2(self, pil_image):
        """converts the given Pillow image to CV2 format"""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def getQualityIndex(self, image):
        original = np.array(self.refImage)
        generated = np.array(image)
        return uqi(original, generated)

    def getMse(self, image):
        """calculates MSE of difference between the given image and the reference image"""
        return np.sum((self.toCv2(image).astype("float") - self.refImageCv2.astype("float")) ** 2)/float(self.numPixels)

    def getCP(self, image):
        original = np.array(self.refImage)
        generated = np.array(image)
        blank = np.zeros(original.shape)

        max_loss = ImageHelper.computeLoss(original, blank)
        best_loss = ImageHelper.computeLoss(original, generated)

        return 100*(max_loss-best_loss)/max_loss

    @staticmethod
    def computeLoss(original: np.ndarray, generated: np.ndarray):
        return np.sum(np.abs(original - generated))

    def getLoss(self, image):
        original = np.array(self.refImage)
        generated = np.array(image)
        return ImageHelper.computeLoss(original, generated)

    def getSsim(self, image):
        """calculates mean structural similarity index between the given image and the reference image"""
        return structural_similarity(self.toCv2(image), self.refImageCv2, multichannel=True)

    def getPSNR(self, image):
        original = self.refImageCv2
        generated = self.toCv2(image)

        mse = np.mean((original - generated) ** 2)
        if (mse == 0):  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    def list2Chunks(self, list, chunkSize):
        """divides a given list to fixed size chunks, returns a generator iterator"""
        for chunk in range(0, len(list), chunkSize):
            yield (list[chunk:chunk + chunkSize])

    def ticksOff(self, plot):  # TODO
        """turns off ticks on both axes"""
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            top=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    def toRealMatrix(self):
        return ImageHelper.ToRealMatrix(self.refImage)

    def ToRealMatrix(image: Image.Image):
        numpydata = np.asarray(image)
        matrix = np.zeros((numpydata.shape[0], numpydata.shape[1]))
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                matrix[r, c] = ImageHelper.GetDoublefromRGB(numpydata[r, c, :])
        return matrix

    def FromRealMatrix(matrix) -> Image.Image:
        data = np.zeros((matrix.shape[0], matrix.shape[1], 3), np.uint8)
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                data[r, c, :] = ImageHelper.GetRGBfromDouble(matrix[r, c])
        image = Image.fromarray(data, mode='RGB')
        return image

    def GetRGBfromDouble(RGBDouble):
        RGBint = int(RGBDouble*16777215)
        # red,gree,blue
        return np.uint8([(RGBint >> 16) & 255, (RGBint >> 8) & 255, RGBint & 255])

    def GetDoublefromRGB(rgb):
        # red,gree,blue
        return ((rgb[0] << 16) + (rgb[1] << 8) + rgb[2])/16777215
