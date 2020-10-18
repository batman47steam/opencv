#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;
static void help(char** argv)
{
	cout << endl
		<< "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
		<< "The dft of an image is taken and it's power spectrum is displayed." << endl << endl
		<< "Usage:" << endl
		<< argv[0] << " [image_name -- default lena.jpg]" << endl << endl;
}
int main(int argc, char** argv)
{
	help(argv);
	const char* filename = argc >= 2 ? argv[1] : "lena.jpg";
	Mat I = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
	if (I.empty()) {
		cout << "Error opening image" << endl;
		return EXIT_FAILURE;
	}


	/*
		先通过getOptimalDFTSize得到优化的行数和列数也就是m和n
		然后把这些多出来的行和列统一设计为0
	*/
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	/*
	1.	频域的运算结果通常要更大，我们先把输入图像转换为float类型
		然后再添加另一个channel，这个channel先设置为全是float类型的
		0元素
	2.	通过merge来把planes[0]和planes[1]合并到一起，用来存放傅里叶
		滤波后的complexI矩阵
	*/

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
	dft(complexI, complexI);            // this way the result may fit in the source matrix

	/*
	1.	利用split把2通道的complexI分开装到planes里
		一个对应实部一个对应虚部
	2.	magnitude利用planes[0]和planes[1]算出复数的模长，然后放入到planes[0]中
	*/
	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];


	/*
		因为傅里也系数的动态范围太大了，所以利用log把这些范围缩小到
		能用灰度图表示的范围中
	*/
	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);


	/*
		-2的补码为11111110，cols & -2相当于把最后一位置零，取小于等于col且最近的偶数
	*/
	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	// rearrange the quadrants of Fourier image  so that the origin is at the image center

	//cx,cy对应图像的center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;
	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//在32位浮点数类型的图像中，它的取值范围是[0,1]，在使用imshow函数来显示图像时，将为我们自动乘以255，以扩展到[0,255]
	normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).
	imshow("Input Image", I);    // Show the result
	imshow("spectrum magnitude", magI);
	waitKey();
	return EXIT_SUCCESS;
}