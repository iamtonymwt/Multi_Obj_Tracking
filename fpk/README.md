# fpk

Store restructured CenterNet related AI input model and fpks

## Introduction
This project includes the Muti-person tracking model based on CenterNet. This repo contains SDSP converted fpk file and input&output tensor dimension info.

<center>
<figure class='half'>
<img width ='100', height = '250' src=./images/model_structure.png/> &emsp;
<img width ='300', height = '250' src=./images/static_demo.png/>;
</figure>
</center>
<center> <font size = 2> model tail and prediction effect </font> </center>
<br/>

PS: actually even original float32 output AI model, SDSP would quantized output to same acount of int8 value automatically!
