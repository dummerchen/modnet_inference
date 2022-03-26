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
	std::cout << "output name:" << output_name << std::endl;
}

cv::Mat HumanSeg::normalize(cv::Mat &image) {
	std::vector<cv::Mat> channels, normalized_image;
	// image分离到vector
	cv::split(image, channels);

	cv::Mat r, g, b;
	b = channels.at(0);
	g = channels.at(1);
	r = channels.at(2);
	// (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
	// -1~1
	b = (b / 255. - 0.5) / 0.5;
	g = (g / 255. - 0.5) / 0.5;
	r = (r / 255. - 0.5) / 0.5;
	//r = (r / 255. - normalized_param[0][0]) / normalized_param[1][0];
	//g = (g / 255. - normalized_param[0][1]) / normalized_param[1][1];
	//b = (b / 255. - normalized_param[0][2]) / normalized_param[1][2];

	normalized_image.push_back(r);
	normalized_image.push_back(g);
	normalized_image.push_back(b);

	cv::Mat normal_image;// 合成图片
	cv::merge(normalized_image,normal_image);
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

	// 图像还是出bug了，有时间能改就改
	rh = rw = refsize;
	cv::Mat resized_image,resized_image_float,normalized_image;
	cv::resize(image, resized_image, cv::Size(rw,rh),0,0,cv::INTER_AREA);
	resized_image.convertTo(resized_image_float, CV_32F);
	normalized_image = normalize(resized_image_float);
	input_node_dims = { 1,3,rh,rw };
	return { resized_image,normalized_image };
}
void addAlpha(cv::Mat& src, cv::Mat& dst, cv::Mat& alpha)
{
	if (src.channels() == 4)
	{
		return ;
	}
	else if (src.channels() == 1)
	{
		cv::cvtColor(src, src, cv::COLOR_GRAY2RGB);
	}

	dst = cv::Mat(src.rows, src.cols, CV_8UC4);

	std::vector<cv::Mat> srcChannels;
	std::vector<cv::Mat> dstChannels;
	//分离通道
	cv::split(src, srcChannels);

	dstChannels.push_back(srcChannels[0]);
	dstChannels.push_back(srcChannels[1]);
	dstChannels.push_back(srcChannels[2]);
	//添加透明度通道
	dstChannels.push_back(alpha);
	//合并通道
	cv::merge(dstChannels, dst);

	return ;
}
/*
* postprocess: preprocessed image -> infer -> postprocess
*/
vector<cv::Mat > HumanSeg::predict(const string& src_path, const string& dst_path,const string image_or_video) 
{
	//cout<<"!!" << endl;
	cv::Mat image = cv::imread(src_path);
	cv::Mat preprocessed_image, resized_image;
	preprocessed_image= preprocess(image)[1];
	resized_image = preprocess(image)[0];

	cv::Mat blob = cv::dnn::blobFromImage(preprocessed_image, 1, cv::Size((preprocessed_image.rows), int(preprocessed_image.cols)), cv::Scalar(0, 0, 0), false, true);
	
	
	// create input tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_node_dims.data(), input_node_dims.size()));

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
	mask_1f *= 255;
	mask_1f.convertTo(mask, CV_8U);
	//cout<<"mask suc" << endl;
	
	cv::Mat pre_image,without_bg;
	// mask不为0的地方全变黑

	//cv::imwrite(dst_path, mask);
	addAlpha(resized_image,pre_image, mask);
	cv::resize(pre_image, without_bg, cv::Size(image_w, image_h), 0, 0, cv::INTER_AREA);

	//std::cout << "predict image over" << std::endl;
	input_tensors.clear();
	return {without_bg,mask};
}
