#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

cv::Mat ideal_Low_Pass_Filter(Mat& src, float sigma);
Mat ideal_lbrf_kernel(Mat& scr, float sigma);
Mat freqfilt(Mat& scr, Mat& blur);

int main(int argc, char* argv[])
{
	const char* filename = argc >= 2 ? argv[1] : "../data/lena.jpg";

	Mat input = imread(filename, IMREAD_GRAYSCALE);
	if (input.empty())
		return -1;
	imshow("input", input);//Show original image

	cv::Mat ideal = ideal_Low_Pass_Filter(input, 160);//160
	ideal = ideal(cv::Rect(0, 0, input.cols, input.rows));
	imshow("ideal", ideal);
	waitKey();
	return 0;
}

//*****************Ideal low pass filter***********************
Mat ideal_lbrf_kernel(Mat& scr, float sigma)
{
	Mat ideal_low_pass(scr.size(), CV_32FC1); //��CV_32FC1
	float d0 = sigma;//The smaller the radius D0, the larger the blur; the larger the radius D0, the smaller the blur
	for (int i = 0; i < scr.rows; i++) {
		for (int j = 0; j < scr.cols; j++) {
			//����ÿ�����ص㵽srcͼ�����ĵľ��룬���С�ڰ뾶������Ϊ0���ȵȣ��Ӷ��γ���һ����ͨ�˲������϶����þ�������Ҫ�úܶ�
			double d = sqrt(pow((i - scr.rows / 2), 2) + pow((j - scr.cols / 2), 2));//Molecule, Calculate pow must be float type
			if (d <= d0) {
				ideal_low_pass.at<float>(i, j) = 1;
			}
			else {
				ideal_low_pass.at<float>(i, j) = 0;
			}
		}
	}
	string name = "ideal low pass filter d0=" + std::to_string(sigma);
	imshow(name, ideal_low_pass);
	return ideal_low_pass;//���ص���һ�����󣬳������ĵ�Բ����ֵΪ1������������ֵ��Ϊ0
}

cv::Mat ideal_Low_Pass_Filter(Mat& src, float sigma)
{
	int M = getOptimalDFTSize(src.rows);
	int N = getOptimalDFTSize(src.cols);
	Mat padded; //Adjust image acceleration Fourier transform
	copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols, BORDER_CONSTANT, Scalar::all(0));
	padded.convertTo(padded, CV_32FC1); //convert the image to float

	Mat ideal_kernel = ideal_lbrf_kernel(padded, sigma);//ideal low pass filter�����������һ����ͨ�˲�����mask
	Mat result = freqfilt(padded, ideal_kernel);
	return result;
}
//***************** Frequency Domain Filtering****************** *
Mat freqfilt(Mat& scr, Mat& blur)
{
	//***********************DFT*******************
	Mat plane[] = { scr, Mat::zeros(scr.size() , CV_32FC1) }; //Create a channel and store the real and imaginary parts after dft (CV_32F, Must be a single channel number)
	Mat complexIm;
	merge(plane, 2, complexIm);//Merge channel (combines two matrices into a 2-channel Mat class container)
	dft(complexIm, complexIm);//Fourier transform, the result is saved in itself��complexIm����ֱ�ӵĸ���Ҷ�任�Ľ�����˿϶��ǿ�������

	//***************Centralization********************
	split(complexIm, plane);//separation channel (array separation)
	// plane[0] = plane[0](Rect(0, 0, plane[0].cols & -2, plane[0].rows & -2)); / / Here why and on -2 specific view opencv documentation
	// //In fact, it is to make the rows and columns into even numbers. The binary of 2 is 11111111.......10 The last bit is 0.
	int cx = plane[0].cols / 2; int cy = plane[0].rows / 2;//The following operations are moving images (zero frequency shift to center)
	Mat part1_r(plane[0], Rect(0, 0, cx, cy)); //Element coordinates are expressed as (cx,cy)
	Mat part2_r(plane[0], Rect(cx, 0, cx, cy));
	Mat part3_r(plane[0], Rect(0, cy, cx, cy));
	Mat part4_r(plane[0], Rect(cx, cy, cx, cy));

	Mat temp;
	part1_r.copyTo(temp); //Upper left and lower right exchange position (real part)
	part4_r.copyTo(part1_r);
	temp.copyTo(part4_r);

	part2_r.copyTo(temp); //Upper right and bottom left exchange position (real part)
	part3_r.copyTo(part2_r);
	temp.copyTo(part3_r);

	Mat part1_i(plane[1], Rect(0, 0, cx, cy)); //Element coordinates (cx, cy)
	Mat part2_i(plane[1], Rect(cx, 0, cx, cy));
	Mat part3_i(plane[1], Rect(0, cy, cx, cy));
	Mat part4_i(plane[1], Rect(cx, cy, cx, cy));

	part1_i.copyTo(temp); //Upper left and lower right exchange position (imaginary part)
	part4_i.copyTo(part1_i);
	temp.copyTo(part4_i);

	part2_i.copyTo(temp); // upper right and lower left exchange position (imaginary part)
	part3_i.copyTo(part2_i);
	temp.copyTo(part3_i);

	//***************** The product of the filter function and the DFT result ******************
	Mat blur_r, blur_i, BLUR;
	//planeӦ�����Ѿ����Ļ����Ժ�ĸ���Ҷ�任�Ľ��
	//multiply������������Ķ�ӦԪ�����
	//blur�Ǵ��������ú��˵��Ǹ�mask
	multiply(plane[0], blur, blur_r); //filter (the real part is multiplied by the corresponding element of the filter template)
	multiply(plane[1], blur, blur_i); // filter (imaginary part is multiplied by the corresponding element of the filter template)
	Mat plane1[] = { blur_r, blur_i };
	merge(plane1, 2, BLUR);//The real and imaginary parts merge �ϳ�һ��˫ͨ����BLUR

	//*********************Get the original spectrum map********************** *************
	magnitude(plane[0], plane[1], plane[0]);//Get the amplitude image, 0 channel is the real channel, 1 is the imaginary part, because 2D Fourier The result of the transformation is a complex number
	plane[0] += Scalar::all(1); //The image after Fourier transform is not well analyzed, and the logarithmic processing is performed. The result is better.
	log(plane[0], plane[0]); // The gray space of the float type is [0,1])
	normalize(plane[0], plane[0], 1, 0, NORM_MINMAX); //normalized for easy display
	imshow("original image spectrogram", plane[0]);

	idft(BLUR, BLUR); //idft result is also plural
	split(BLUR, plane);//Separate channel, mainly get channel
	magnitude(plane[0], plane[1], plane[0]); //Amplitude (modulo)
	normalize(plane[0], plane[0], 1, 0, NORM_MINMAX); //normalized for easy display
	return plane[0];//return parameters
}