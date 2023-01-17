#需要分步进行
#convpy -i /mwt_workspace/MultiObjTracking/SDSP/sdsp/size8_modified.tflite -o convpy_output/
#DnnConv -n convpy_output/ -o dnnconv_output/
#ConvBE -f dnnconv_output/convpy_output -o convbe_output/

#IMX500 packager commands (optional)
#unzip convbe_output/packerOut.zip -d ../../../../workspace/SDSP_Packager/IMX500_packager_v2_0_1/CustomNetwork/packer_output/
cd ../../../../workspace/SDSP_Packager/IMX500_packager_v2_0_1/CustomNetwork/
./mk_network_pkg_sample.sh
