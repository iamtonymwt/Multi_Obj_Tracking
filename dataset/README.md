# instructions
## raw dataset
* /caltech/annotations 
    * set**
        * *.vbb
* /caltech/set**
    * V***.seq
## python files
* visualize.py
    * used for visualizing three demo images
* firstly use ann_convert.py to convert annotations to xml, then use img_convert.py to change images to .jpg. 
* Because in severals pictures, there might be no people or they are really small. So we use ann_select.py to select usable annotations.
* in order to match the annotations and images, we use img_select.py to select images.
* finally, we could use genList.py to generate the info list