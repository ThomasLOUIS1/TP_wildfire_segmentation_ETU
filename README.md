<H1 align="center" style="color:darkviolet"> SENTINEL2 FOREST FIRES DETECTION </H1>

<p align="center">
<img src="images/logo.png"  width=300" height="200"><br>
</p>

## Educational Notebook to train a deep learning model to detect forest fires using sentinel-2 multi-spectral satellite imagery


<p align="center">
<img src="images/predictions.jpg" ><br>
</p>

**Multi-spectral images** are images representing the same area in **multiple wavelength band**. They are very useful for wildfire segmentation because they can provide more information about the characteristics of the wildfire and the surrounding area than RGB images. 

For example, near-infrared (NIR) bands can be used to identify vegetation, while shortwave infrared (SWIR) bands can be used to detect the presence of smoke. Additionally, using multiple wavelength bands can help to reduce the impact of atmospheric conditions such as clouds and haze on the image. This **can improve the accuracy of the segmentation task** and make it easier to **identify a wildfire** in an image.

These images are stored in `.tif files`. **TIFF** is a widely-used **file format for images**. It is capable of storing images in a lossless format, meaning that **no data is lost when the image is compressed**. This makes it a popular choice for storing **high-quality images**, such as those used in professional photography, printing or satellite imagery.
                                 
Above are four **256x256** images :
 - RGB images where each pixel is an 8-bit unsigned integer (uint8)
 - The reconstructed images from the 3 spectral bands in uint8 (named "False Color")
 - The binary masks of segmentation (named "GT Mask"). This is an "image" of the same size as the original reconstructed image but with values 0 and 1 (**Black/1 = no-fire; White/0 = fire**)
 - Some predictions performed by a neural network (named "Prediction").


<div id="top"></div>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-this-repository">About this repository</a></li>
    <li>
      <a href="#getting-Started">Getting started</a>
    </li>
    <li>
        <a href="#usage">Usage</a>
        <ul>
            <li><a href="#Notebook_running">Notebook running</a></li>
            <li><a href="#data_structure">Data structure</a></li>
        </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <!-- <li><a href="#references">References</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## <div id="about-this-repository">1. About this repository 💼 </div>

This git repository contains a Notebook to perform forest fire detection using deep learning. 
The Notebook consists in loading the dataset, creating several models trying to get good results and making the inference on some sample images.


### <div id="packages"> This Notebook is built with </div>

* [Anaconda](https://www.anaconda.com/products/distribution)
* [Python](https://www.python.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Matplotlib](https://matplotlib.org/)
* [NumPy](https://numpy.org/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->

## <div id="getting-Started"> 2.1 Getting Started (Colab)📚  </div>

* Open this link in a new tab (CTRL+Click or Middle_click): [Run colab from this github](https://githubtocolab.com/ThomasLOUIS1/TP_wildfire_segmentation_ETU.git)

* Choose : 
    - Branche : main
    - Click "TP_wildfire.ipynb"

* Then :
    - :uk: go to Runtime >> Change runtime type >> Choose GPU
    - :fr: go to Execution >> Modifier le type d'éxecution >> Choose GPU


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->


## <div id="license">4. License 📑</div>

Distributed under the Attribution 4.0 International (CC BY 4.0) License. 

Contains modified Copernicus Sentinel data [2016-2020] for Sentinel data

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->


## <div id="contact">5. Contact 📭</div>

* [Thomas LOUIS 📧](mailto:thomas.louis@irt-saintexupery.com)
* [Alain PEGATOQUET 📧](mailto:alain.pegatoquet@univ-cotedazur.fr)



Project Link: [https://github.com/ThomasLOUIS1/TP_wildfire_segmentation_ETU](https://github.com/ThomasLOUIS1/TP_wildfire_segmentation_ETU)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- REFERENCES -->

<!-- ## <div id="references">5. References 📭 </div>


<p align="right">(<a href="#top">back to top</a>)</p> -->

