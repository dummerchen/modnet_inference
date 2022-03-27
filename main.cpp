#include<bits/stdc++.h>
#include<iostream>
#include<opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<gflags\gflags.h>
#include<codecvt>
#include "humanseg.h"
using namespace std;
using namespace cv;
// name.exe onnx_path 

DEFINE_string(onnx_path,"./onnx/modnet_webcam_portrait_matting.onnx","model path");
//DEFINE_string(onnx_path, "./onnx/modnet.onnx", "model path");
//DEFINE_string(test_path, "E:/py_exercise/service_project/ccodes/test/image/0.png", "test path");

DEFINE_string(test_path, "E:/py_exercise/service_project/datasets/test/TEST_04.mp4", "test path");
DEFINE_int32(num_thread,7, "threads nums");

vector <string> split_name(string path)
{
	size_t pos = path.find_last_of('/');
	string name;
	if (pos != NULL)
	{
		name= path.substr(pos + 1);
	}
	else
	{
		size_t pos = path.find_last_of('\\');
		name=path.substr(pos + 1);
	}
	pos = name.find_last_of('.');
	string suffix = name.substr(pos + 1);
	return vector<string>{name, suffix};
}

int main(int argc, char** argv)
{
	google::ParseCommandLineFlags(&argc, &argv, true);
	wstring_convert < codecvt_utf8_utf16<wchar_t> > converter;
	cout << FLAGS_num_thread << FLAGS_onnx_path << FLAGS_test_path << endl;
	wstring path = converter.from_bytes(FLAGS_onnx_path);

	// model init
	HumanSeg humanseg(path, FLAGS_num_thread);

	vector<string> test_info = split_name(FLAGS_test_path);
	string image_or_video, suffix, name;
	cv::Mat res;

	name = test_info[0];
	suffix = test_info[1];
	if ((suffix == "jpg") || (suffix == "png"))
	{
		image_or_video = "image";
		cv::Mat image = cv::imread(FLAGS_test_path);
		res = humanseg.predict(image, image_or_video);
		cv::imwrite("./result/" + image_or_video + "/without_bg_" + test_info[0], res);
		//cv::imwrite("./result/" + image_or_video + "/mask_" + test_info[0], res[1]);
		image.release();
	}
	else
	{
		image_or_video = "video";
		cv::VideoCapture capture;
		
		cv::Mat frame,image;
		capture.open(FLAGS_test_path);
		if(capture.isOpened())
			cout<<"video open success"<<endl;
		else
			cout<<"video open failed"<<endl;

		Size S = Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH),(int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
		int fps = capture.get(cv::CAP_PROP_FPS);
		cv::VideoWriter writer("./result/"+image_or_video+"/" + test_info[0], VideoWriter::fourcc('M', 'P', '4', 'V'), fps, S, true);
		while (capture.read(frame))
		{
			//frame.copyTo(image);

			clock_t start_time = clock();

			res=humanseg.predict(frame, image_or_video);

			clock_t end_time = clock();
			std::cout << (end_time - start_time) / 1000.0 << std::endl;
			writer.write(res);
			//cv::imshow("original", frame);
			//cv::imshow("Result", res[0]);
		}
		capture.release();
		writer.release();
	}

	
	google::ShutDownCommandLineFlags();
	
	cv::waitKey(0);
	//system("pause");
	return 0;
}