# image processing

#### 介绍
简单的opencv图像处理，用于二维码识别

#### 步骤、

opencv采用vcpkg安装管理，版本为4.5.1

~~~
git clone https://github.com/Microsoft/vcpkg.git

###cd安装目录
.\vcpkg\bootstrap-vcpkg.bat
.\vcpkg install opencv:x64-windows
~~~

ZBar压缩包为条码、二维码识别库，解压即可

test.png为测试图片，复制到可执行文件同级目录

修改CMakeLists.list中Zbar安装目录，vcpkg安装目录

编译运行

