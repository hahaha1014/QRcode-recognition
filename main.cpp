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
using namespace zbar;  //���zbar���ƿռ�    
using namespace cv;

RNG rng(12345);
//Scalar colorful = CV_RGB(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));

/*********ͼƬ��Ȩƽ���ҶȻ�*********/
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
			data1[x] = (int)(R * 0.3 + G * 0.59 + B * 0.11);//���ù�ʽ����Ҷ�ֵ����Ȩƽ������
		}
	}
	return dst;
}

/*********��ֵ�˲�����-�����ٶȽ���**MedianFilter(imageGray, imagefilter, wsize);*******/
//ð�����򣬴�С
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
	//ɸѡ��λͼ�Σ�������Ƕ������
	int parentIdx = -1;
	for (int i = 0; i < contours.size(); i++)
	{
		if (hierarchy[i][2] != -1 && ic == 0)
		{
			parentIdx = i;
			ic++;
		}
		else if (hierarchy[i][2] != -1)//��������
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

float vector_cross_mul(Point2f& const point, Point2f& const point1, Point2f& const point2) {//�������
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


/*********�ҵ�����ȡ���������ĵ�*********/
Point Center_cal(vector<vector<Point> > contours, int i)
{
	int centerx = 0, centery = 0, n = contours[i].size();
	//����ȡ��С�����εı߽���ÿ���ܳ���������ȡһ��������꣬������ȡ�ĸ����ƽ�����꣨��ΪС�����εĴ������ģ�
	centerx = (contours[i][n / 4].x + contours[i][n * 2 / 4].x + contours[i][3 * n / 4].x + contours[i][n - 1].x) / 4;
	centery = (contours[i][n / 4].y + contours[i][n * 2 / 4].y + contours[i][3 * n / 4].y + contours[i][n - 1].y) / 4;
	Point point1 = Point(centerx, centery);
	return point1;
}

int main(int argc, char*argv[])
{

	//����ɨ����
	ImageScanner scanner;
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

	//1����ͼƬ
	Mat image = imread("./test.png");//�����ͼƬ������ʹ�þݶ�·�������·��
	if (!image.data)
	{
		cout << "��ȷ��ͼƬ" << endl;
		system("pause");
		return 0;
	}
	clock_t start, end;
	start = clock();//time test
	resize(image, image, Size(2560, 1920));//640,480
	namedWindow("1ԭͼ", WINDOW_NORMAL);
	resizeWindow("1ԭͼ", 640, 480);//���
	imshow("1ԭͼ", image);
	Mat imageCopy;
	image.copyTo(imageCopy);//ͼ����

	//2�ҶȻ�
	Mat imageGray;
	imageGray = Color2Gray(image);
	//cvtColor(image, imageGray, COLOR_BGR2GRAY); //ͼ��ҶȻ�
	namedWindow("2�ҶȻ�", WINDOW_NORMAL);
	resizeWindow("2�ҶȻ�", 640, 480);
	imshow("2�ҶȻ�", imageGray);

	//3�˲�
	Mat imageFilter;
	//Size wsize(5, 5);
	//MedianFilter(imageGray, imagefilter, wsize); //��ֵ�˲�
	medianBlur(imageGray, imageFilter, 7);
	namedWindow("3�˲�", WINDOW_NORMAL);
	resizeWindow("3�˲�", 640, 480);
	imshow("3�˲�", imageFilter);

	//4��ֵ��
	Mat imageOTSU;
	//threshold(imageFilter, imageOTSU, 0, 255, THRESH_OTSU);//OTSU
	adaptiveThreshold(imageFilter, imageOTSU, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 235, 8);//����Ӧ
	namedWindow("4��ֵ��", WINDOW_NORMAL);
	resizeWindow("4��ֵ��", 640, 480);
	imshow("4��ֵ��", imageOTSU);

	//5��˹�˲�
	Mat imageGauss;
	GaussianBlur(imageOTSU, imageGauss, Size(11, 11), 0, 0);
	namedWindow("5��˹�˲�", WINDOW_NORMAL);
	resizeWindow("5��˹�˲�", 640, 480);
	imshow("5��˹�˲�", imageGauss);

	//6sobel�������Ե
	Mat imageSobel;
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
	Sobel(imageGauss, grad_x, CV_16S, 1, 0, 3);        // use CV_16S to avoid overflow
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(imageGauss, grad_y, CV_16S, 0, 1, 3);        // use CV_16S to avoid overflow
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imageSobel);
	namedWindow("6sobel��Ե", WINDOW_NORMAL);
	resizeWindow("6sobel��Ե", 640, 480);
	imshow("6sobel��Ե", imageSobel);
	
	//7Ѱ�Ҷ�λ��
	Scalar color = Scalar(0, 0, 255);
	vector<vector<Point> > contours, contours2;
	vector<Vec4i> hierarchy;
	//Mat drawing = Mat::zeros(image.size(), CV_8UC3);
	//Ѱ������ 
	//��һ������������ͼ�� 2ֵ����
	//�ڶ����������ڴ�洢����FindContours�ҵ��������ŵ��ڴ����档
	//�����������ǲ㼶��**[Next, Previous, First_Child, Parent]** ��vector
	//���ĸ����������ͣ��������ṹ
	//����������ǽڵ����ģʽ��������ȫ��Ѱ��
	findContours(imageSobel, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

	corner_find(contours, hierarchy, 5, contours2);//��������λ��
	double area = contourArea(contours2[1]);
	int area_side = cvRound(sqrt(area));
	for (int i = 0; i < contours2.size(); i++) {
		drawContours(imageCopy, contours2, i, color, area_side / 8, 4, hierarchy[0][2], 0, Point());//����λ��
	}
	
	Point2f corner_center[3];
	Point2f center;//���ĵ�����
	vector<int> distance;
	double distance_max = 0;
	int cornerA;//���϶�λ��
	double theta = 0, rotate_theta = 0;//��ת��
	for (int i = 0; i < contours2.size(); i++)
	{
		corner_center[i] = Center_cal(contours2, i);//��λ�ǵ����ĵ�
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
	namedWindow("7��λ��", WINDOW_NORMAL);
	resizeWindow("7��λ��", 640, 480);
	imshow("7��λ��", imageCopy);
	
	//8����ά������
	Point2f contour_point[3][4];
	Point2f corner_point[3][2];
	float vector_length, vector_length_max = 0 , vector_length_min = 0;
	for (int i = 0; i < contours2.size(); i++) {
		RotatedRect rectPoint = minAreaRect(contours2[i]);//����С��Χ���Σ�б��Ҳ����
		rectPoint.points(contour_point[i]);//��rectPoint�����д洢������ֵ�ŵ� fourPoint��������

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
		corner_point[(cornerA + 2) % 3][0], corner_point[(cornerA + 2) % 3][1]);//������ĸ���λ��
	//pointD = { 520,1450 };
	//�жϽǵ����
	Point2f fourPoint[4];
	float vector_cross = vector_cross_mul(center, contour_point[(cornerA + 1) % 3][0], corner_center[cornerA]);
	if (vector_cross > 0) {//ע��y�Ƿ����µ�
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
		line(image, fourPoint[i % 4], fourPoint[(i + 1) % 4], Scalar(20, 21, 255), 20);//�����Ͻ�Ϊԭ�㣬��x��y
	}
	namedWindow("8������", WINDOW_NORMAL);
	resizeWindow("8������", 640, 480);
	imshow("8������", image);

	//9͸�ӱ任�����Բ�ֵ
	Point2f fourPoint2[4];
	fourPoint2[0] = Point2f(100, 100);
	fourPoint2[1] = Point2f(1800, 100);
	fourPoint2[2] = Point2f(1800, 1800);
	fourPoint2[3] = Point2f(100, 1800);
	Mat trans = getPerspectiveTransform(fourPoint, fourPoint2);
	Mat imageTrans;
	warpPerspective(imageOTSU, imageTrans, trans, Size(image.cols, image.rows), INTER_LINEAR);//͸�ӱ任�����Բ�ֵ
	namedWindow("9У����", WINDOW_NORMAL);
	resizeWindow("9У����", 640, 480);
	imshow("9У����", imageTrans);

	int width = imageTrans.cols;
	int height = imageTrans.rows;
	uchar *raw = (uchar *)imageTrans.data;
	Image imageZbar(width, height, "Y800", raw, width * height);
	scanner.scan(imageZbar); //ɨ������    
	Image::SymbolIterator symbol = imageZbar.symbol_begin();

	//ɨ������ӡ
	if (imageZbar.symbol_begin() == imageZbar.symbol_end())
	{
		cout << "��ѯ����ʧ�ܣ�����ͼƬ��" << endl;
	}
	for (; symbol != imageZbar.symbol_end(); ++symbol)
	{
		cout << "�������ͣ�" << endl << symbol->get_type_name() << endl << endl;
		cout << "�������ݣ�" << endl << symbol->get_data() << endl << endl;
	}
	end = clock();
	cout << "time = " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
	waitKey();
	imageZbar.set_data(NULL, 0);//�������
	return 0;
}


