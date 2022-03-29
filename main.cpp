#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <zbar.h>  
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265

using namespace std;
using namespace zbar;  //添加zbar名称空间    
using namespace cv;

RNG rng(12345);
//Scalar colorful = CV_RGB(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));

/*********图片加权平均灰度化*********/
Mat Color2Gray(Mat& const src) {
	Mat dst;
	dst = Mat(src.rows, src.cols, CV_8UC1);
	float R, G, B;
	for (int y = 0; y < src.rows; y++)
	{
		uchar* data1 = dst.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++)
		{
			B = src.at<Vec3b>(y, x)[0];
			G = src.at<Vec3b>(y, x)[1];
			R = src.at<Vec3b>(y, x)[2];
			data1[x] = (int)(R * 0.3 + G * 0.59 + B * 0.11);//利用公式计算灰度值（加权平均法）
		}
	}
	return dst;
}

/*********中值滤波函数-运行速度较慢**MedianFilter(imageGray, imagefilter, wsize);*******/
//冒泡排序，大到小
void bublle_sort(vector<int> &arr) {
	bool flag = true;
	for (int i = 0; i < arr.size() - 1; ++i) {
		while (flag) {
			flag = false;
			for (int j = 0; j < arr.size() - 1 - i; ++j) {
				if (arr[j] < arr[j + 1]) {
					int tmp = arr[j];
					arr[j] = arr[j + 1];
					arr[j + 1] = tmp;
					flag = true;
				}
			}
		}
	}
}

void corner_find(vector<vector<Point> >& const contours, vector<Vec4i>& hierarchy, const int count, vector<vector<Point> > &contours2) {
	int ic = 0;
	//筛选定位图形，有三层嵌套轮廓
	int parentIdx = -1;
	for (int i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][2] != -1 && ic == 0)
		{
			parentIdx = i;
			ic++;
		}
		else if (hierarchy[i][2] != -1)//有子轮廓
		{
			ic++;
		}
		else if (hierarchy[i][2] == -1)
		{
			ic = 0;
			parentIdx = -1;
		}
		if (ic == count)
		{
			contours2.push_back(contours[parentIdx]);
			ic = 0;
			parentIdx = -1;
		}
	}
}

float vector_mul(Point2f& const point, Point2f& const point1, Point2f& const point2) {
	float x1 = point1.x - point.x;
	float y1 = point1.y - point.y;
	float x2 = point2.x - point.x;
	float y2 = point2.y - point.y;
	float mul = x1 * x2 + y1 * y2;
	return mul;
}

float vector_cross_mul(Point2f& const point, Point2f& const point1, Point2f& const point2) {//向量叉乘
	float x1 = point1.x - point.x;
	float y1 = point1.y - point.y;
	float x2 = point2.x - point.x;
	float y2 = point2.y - point.y;
	float mul = x1 * y2 - x2 * y1;
	return mul;
}

Point2f GetCrossPoint(Point2f& const p0, Point2f& const p1, Point2f& const p2, Point2f& const p3) {//x-->,y--down
	Point2f point = { 0,0 };
	float a0 = p0.y - p1.y;
	float b0 = p1.x - p0.x;
	float c0 = p0.x*p1.y - p1.x*p0.y;
	float a1 = p2.y - p3.y;
	float b1 = p3.x - p2.x;
	float c1 = p2.x*p3.y - p3.x*p2.y;
	float D = a0 * b1 - a1 * b0;
	point.x = (b0*c1 - b1 * c0) / D;
	point.y = (a1*c0 - a0 * c1) / D;
	return point;
}


/*********找到所提取轮廓的中心点*********/
Point Center_cal(vector<vector<Point> > contours, int i)
{
	int centerx = 0, centery = 0, n = contours[i].size();
	//在提取的小正方形的边界上每隔周长个像素提取一个点的坐标，求所提取四个点的平均坐标（即为小正方形的大致中心）
	centerx = (contours[i][n / 4].x + contours[i][n * 2 / 4].x + contours[i][3 * n / 4].x + contours[i][n - 1].x) / 4;
	centery = (contours[i][n / 4].y + contours[i][n * 2 / 4].y + contours[i][3 * n / 4].y + contours[i][n - 1].y) / 4;
	Point point1 = Point(centerx, centery);
	return point1;
}

int main(int argc, char*argv[])
{

	//定义扫描器
	ImageScanner scanner;
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

	//1加载图片
	Mat image = imread("./test.png");//请更改图片，可以使用据对路径或相对路径
	if (!image.data)
	{
		cout << "请确认图片" << endl;
		system("pause");
		return 0;
	}
	clock_t start, end;
	start = clock();//time test
	resize(image, image, Size(2560, 1920));//640,480
	namedWindow("1原图", WINDOW_NORMAL);
	resizeWindow("1原图", 640, 480);//宽高
	imshow("1原图", image);
	Mat imageCopy;
	image.copyTo(imageCopy);//图像复制

	//2灰度化
	Mat imageGray;
	imageGray = Color2Gray(image);
	//cvtColor(image, imageGray, COLOR_BGR2GRAY); //图像灰度化
	namedWindow("2灰度化", WINDOW_NORMAL);
	resizeWindow("2灰度化", 640, 480);
	imshow("2灰度化", imageGray);

	//3滤波
	Mat imageFilter;
	//Size wsize(5, 5);
	//MedianFilter(imageGray, imagefilter, wsize); //中值滤波
	medianBlur(imageGray, imageFilter, 7);
	namedWindow("3滤波", WINDOW_NORMAL);
	resizeWindow("3滤波", 640, 480);
	imshow("3滤波", imageFilter);

	//4二值化
	Mat imageOTSU;
	//threshold(imageFilter, imageOTSU, 0, 255, THRESH_OTSU);//OTSU
	adaptiveThreshold(imageFilter, imageOTSU, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 235, 8);//自适应
	namedWindow("4二值化", WINDOW_NORMAL);
	resizeWindow("4二值化", 640, 480);
	imshow("4二值化", imageOTSU);

	//5高斯滤波
	Mat imageGauss;
	GaussianBlur(imageOTSU, imageGauss, Size(11, 11), 0, 0);
	namedWindow("5高斯滤波", WINDOW_NORMAL);
	resizeWindow("5高斯滤波", 640, 480);
	imshow("5高斯滤波", imageGauss);

	//6sobel算子求边缘
	Mat imageSobel;
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
	Sobel(imageGauss, grad_x, CV_16S, 1, 0, 3);        // use CV_16S to avoid overflow
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(imageGauss, grad_y, CV_16S, 0, 1, 3);        // use CV_16S to avoid overflow
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imageSobel);
	namedWindow("6sobel边缘", WINDOW_NORMAL);
	resizeWindow("6sobel边缘", 640, 480);
	imshow("6sobel边缘", imageSobel);
	
	//7寻找定位角
	Scalar color = Scalar(0, 0, 255);
	vector<vector<Point> > contours, contours2;
	vector<Vec4i> hierarchy;
	//Mat drawing = Mat::zeros(image.size(), CV_8UC3);
	//寻找轮廓 
	//第一个参数是输入图像 2值化的
	//第二个参数是内存存储器，FindContours找到的轮廓放到内存里面。
	//第三个参数是层级，**[Next, Previous, First_Child, Parent]** 的vector
	//第四个参数是类型，采用树结构
	//第五个参数是节点拟合模式，这里是全部寻找
	findContours(imageSobel, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

	corner_find(contours, hierarchy, 5, contours2);//找三个定位角
	double area = contourArea(contours2[1]);
	int area_side = cvRound(sqrt(area));
	for (int i = 0; i < contours2.size(); i++) {
		drawContours(imageCopy, contours2, i, color, area_side / 8, 4, hierarchy[0][2], 0, Point());//画定位角
	}
	
	Point2f corner_center[3];
	Point2f center;//中心点坐标
	vector<int> distance;
	double distance_max = 0;
	int cornerA;//左上定位角
	double theta = 0, rotate_theta = 0;//旋转角
	for (int i = 0; i < contours2.size(); i++)
	{
		corner_center[i] = Center_cal(contours2, i);//定位角的中心点
	}
	for (int i = 0; i < contours2.size(); i++)
	{
		line(imageCopy, corner_center[i%contours2.size()], corner_center[(i + 1) % contours2.size()], color, area_side / 8, 8);
		double distance_x, distance_y;
		distance_x = corner_center[i%contours2.size()].x - corner_center[(i + 1) % contours2.size()].x;
		distance_y = corner_center[i%contours2.size()].y - corner_center[(i + 1) % contours2.size()].y;
		distance.push_back(sqrt(pow(distance_x,2)+ pow(distance_y, 2)));
		if (distance[i] > distance_max) {
			cornerA = (i + 2) % contours2.size();
			distance_max = distance[i];
			center.x = 0.5*(corner_center[i%contours2.size()].x + corner_center[(i + 1) % contours2.size()].x);
			center.y = 0.5*(corner_center[i%contours2.size()].y + corner_center[(i + 1) % contours2.size()].y);
			double x, y;
			x = corner_center[(i + 2) % contours2.size()].x - center.x;
			y = center.y - corner_center[(i + 2) % contours2.size()].y;
			theta = atan(y / x);
			if (y > 0 && theta < 0)theta += PI;
			if (y < 0 && theta > 0)theta -= PI;
			rotate_theta = theta - 3*PI/4;
		}
	}
	namedWindow("7定位角", WINDOW_NORMAL);
	resizeWindow("7定位角", 640, 480);
	imshow("7定位角", imageCopy);
	
	//8画二维码轮廓
	Point2f contour_point[3][4];
	Point2f corner_point[3][2];
	float vector_length, vector_length_max = 0 , vector_length_min = 0;
	for (int i = 0; i < contours2.size(); i++) {
		RotatedRect rectPoint = minAreaRect(contours2[i]);//求最小包围矩形，斜的也可以
		rectPoint.points(contour_point[i]);//将rectPoint变量中存储的坐标值放到 fourPoint的数组中

	}
	for (int i = 0; i < contours2.size(); i++) {
		vector_length_max = 0;
		for (int j = 0; j < 4; j++) {
			vector_length = vector_mul(center, corner_center[i], contour_point[i][j]);
			line(image, contour_point[i][j%4], contour_point[i][(j+1) % 4], color, area_side / 8, 8);//test
			if (vector_length_max < vector_length) {
				vector_length_max = vector_length;
				corner_point[i][0] = contour_point[i][j];
			}
		}
		if (i == cornerA)corner_point[i][1] = Point2f{ -1,-1 };
		else {
			vector_length_min = vector_mul(center, corner_center[cornerA], contour_point[i][0]);
			for (int j = 0; j < 4; j++) {
				vector_length = vector_mul(center, corner_center[cornerA], contour_point[i][j]);
				if (vector_length_min >= vector_length) {
					vector_length_min = vector_length;
					corner_point[i][1] = contour_point[i][j];
				}
			}
		}
	}

	Point2f pointD = GetCrossPoint(corner_point[(cornerA + 1) % 3][0], corner_point[(cornerA + 1) % 3][1], 
		corner_point[(cornerA + 2) % 3][0], corner_point[(cornerA + 2) % 3][1]);//计算第四个定位点
	//pointD = { 520,1450 };
	//判断角点次序
	Point2f fourPoint[4];
	float vector_cross = vector_cross_mul(center, contour_point[(cornerA + 1) % 3][0], corner_center[cornerA]);
	if (vector_cross > 0) {//注意y是方向朝下的
		fourPoint[0] = corner_point[cornerA][0];
		fourPoint[1] = corner_point[(cornerA + 2) % 3][0];
		fourPoint[2] = pointD; 
		fourPoint[3] = corner_point[(cornerA + 1) % 3][0];
	}
	else {
		fourPoint[0] = corner_point[cornerA][0];
		fourPoint[1] = corner_point[(cornerA + 1) % 3][0];
		fourPoint[2] = pointD;
		fourPoint[3] = corner_point[(cornerA + 2) % 3][0];
	}

	for (int i = 0; i < 4; i++)
	{
		line(image, fourPoint[i % 4], fourPoint[(i + 1) % 4], Scalar(20, 21, 255), 20);//以左上角为原点，横x竖y
	}
	namedWindow("8外轮廓", WINDOW_NORMAL);
	resizeWindow("8外轮廓", 640, 480);
	imshow("8外轮廓", image);

	//9透视变换，线性插值
	Point2f fourPoint2[4];
	fourPoint2[0] = Point2f(100, 100);
	fourPoint2[1] = Point2f(1800, 100);
	fourPoint2[2] = Point2f(1800, 1800);
	fourPoint2[3] = Point2f(100, 1800);
	Mat trans = getPerspectiveTransform(fourPoint, fourPoint2);
	Mat imageTrans;
	warpPerspective(imageOTSU, imageTrans, trans, Size(image.cols, image.rows), INTER_LINEAR);//透视变换，线性插值
	namedWindow("9校正后", WINDOW_NORMAL);
	resizeWindow("9校正后", 640, 480);
	imshow("9校正后", imageTrans);

	int width = imageTrans.cols;
	int height = imageTrans.rows;
	uchar *raw = (uchar *)imageTrans.data;
	Image imageZbar(width, height, "Y800", raw, width * height);
	scanner.scan(imageZbar); //扫描条码    
	Image::SymbolIterator symbol = imageZbar.symbol_begin();

	//扫描结果打印
	if (imageZbar.symbol_begin() == imageZbar.symbol_end())
	{
		cout << "查询条码失败，请检查图片！" << endl;
	}
	for (; symbol != imageZbar.symbol_end(); ++symbol)
	{
		cout << "条码类型：" << endl << symbol->get_type_name() << endl << endl;
		cout << "条码内容：" << endl << symbol->get_data() << endl << endl;
	}
	end = clock();
	cout << "time = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
	waitKey();
	imageZbar.set_data(NULL, 0);//清除缓存
	return 0;
}


