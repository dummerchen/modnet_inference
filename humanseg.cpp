#include "HumanSeg.h"
#include<bits/stdc++.h>
using namespace std;
HumanSeg::HumanSeg(std::wstring model_path, int num_threads = 1) 
{

	//std::cout << input_tensor_size_ << std::endl;
	session_options.SetIntraOpNumThreads(num_threads);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	try {
		session = Ort::Session(env_, model_path.c_str(), session_options);
	}
	catch (Ort::Exception& e) {
		std::cout << e.what() << std::endl;
	}

	Ort::AllocatorWithDefaultOptions allocator;
	//获取输入name
	const char* input_name = session.GetInputName(0, allocator);
	input_node_names = { input_name };
	Ort::TypeInfo info = session.GetInputTypeInfo(0);
	//cout << "input name:" << input_name << "node dim" << info.GetTensorTypeAndShapeInfo().GetShape() << endl;

	// 获取输出name和
	const char* output_name = session.GetOutputName(0, allocator);
	out_node_names = { output_name };
}

cv::Mat HumanSeg::normalize(cv::Mat &image) {
	std::vector<cv::Mat> channels, normalized_image;
	// image分离到vector,rgb格式
	cv::split(image, channels);
	channels[0] = (channels[0] - 0.485) / 0.229;
	channels[1] = (channels[1] - 0.456) / 0.224;
	channels[2] = (channels[2] - 0.406) / 0.225;
	cv::Mat normal_image;
	//normal_image = image;
	//normal_image = (image - 0.5) / 0.5;
	//b = channels.at(0);
	//g = channels.at(1);
	//r = channels.at(2);
	// (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

	// 合成图片
	cv::merge(channels,normal_image); //6ms
	return normal_image;
}

/*
* preprocess: resize -> normalize
*/
vector<cv::Mat> HumanSeg::preprocess(cv::Mat &image) 
{
	
	image_h = image.rows;
	image_w = image.cols;
	int rw, rh;
	/*
	if (image_w > image_h)
	{
		rh = 512;
		rw = (image_w*1.0 / image_h) *refsize;
	}
	else
	{
		rw = 512;
		rh = (image_h *1.0/ image_w) * refsize;
	}
	rh -= rh % 32;
	rw -= rw % 32;
	*/

	// 图像还是出bug了，有时间能改就改
	rh = rw = refsize;
	cv::Mat resized_image,resized_image_float,normalized_image;
	cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);
	cv::resize(image, resized_image, cv::Size(rw,rh),0,0,cv::INTER_AREA); //41ms
	
	resized_image.convertTo(resized_image_float, CV_32F,1.0/255); //14 ms 
	normalized_image = normalize(resized_image_float); // 16ms
	input_node_dims = { 1,3,rh,rw };
	return { resized_image,normalized_image };
}


cv::Mat add_alpha(cv::Mat image, cv::Mat mask)
{
	cv::Mat channels[3];
	cv::split(image, channels);
	for (int i = 0; i < 512; i++)
	{
		for (int j = 0; j < 512; j++)
		{
			float m = mask.at<float>(i, j);
			image.at<cv::Vec3b>(i, j)[0] = image.at<cv::Vec3b>(i, j)[0] * m + (1 - m)*255;
			image.at<cv::Vec3b>(i, j)[1] = image.at<cv::Vec3b>(i, j)[1] * m + (1 - m) * 255;
			image.at<cv::Vec3b>(i, j)[2] = image.at<cv::Vec3b>(i, j)[2] * m + (1 - m) * 255;

		}
	}
	return image;
}
/*
* postprocess: preprocessed image -> infer -> postprocess
*/
cv::Mat HumanSeg::predict(cv::Mat &image,const string image_or_video) 
{
	//cout<<"!!" << endl;
	
	cv::Mat normalized_image, resized_image;
	normalized_image= preprocess(image)[1];
	resized_image = preprocess(image)[0];

	// 14ms
	//cv::Mat blob = cv::dnn::blobFromImage(preprocessed_image, 1, cv::Size((preprocessed_image.rows), (preprocessed_image.cols)), cv::Scalar(0, 0, 0), false, true);
	cv::Mat blob = cv::dnn::blobFromImage(normalized_image, 1, cv::Size((512),(512)), cv::Scalar(0, 0, 0), false, true);

	
	// create input tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_node_dims.data(), input_node_dims.size()));
	
	// 798ms
	std::vector<Ort::Value> output_tensors = session.Run(
		Ort::RunOptions{ nullptr },
		input_node_names.data(),
		input_tensors.data(),
		input_node_names.size(),
		out_node_names.data(),
		out_node_names.size()
	);
	// h,w,3

	float* floatarr = output_tensors[0].GetTensorMutableData<float>();
	cv::Mat mask, mask_1f;
	mask_1f = cv::Mat1f(input_node_dims[2],input_node_dims[3], floatarr);
	//mask_1f *= 255.0;
	//mask_1f.convertTo(mask, CV_8U);
	
	cv::Mat pre_image,without_bg,mask_thresh;
	pre_image=add_alpha(resized_image, mask_1f);

	//cv::threshold(mask, mask_thresh, 250, 255, cv::THRESH_BINARY);
	//cv::bitwise_and(resized_image,resized_image, pre_image, mask_thresh);
	
	cv::resize(pre_image, without_bg, cv::Size(image_w, image_h), 0, 0, cv::INTER_AREA);// 123ms
	input_tensors.clear();
	return without_bg;
}
