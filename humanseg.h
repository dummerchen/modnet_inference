#include <string>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

class HumanSeg
{
protected:
	Ort::Env env_;
	Ort::SessionOptions session_options;
	Ort::Session session{nullptr};
	Ort::RunOptions run_options;

	std::vector<Ort::Value> input_tensors;


	std::vector<const char*> input_node_names;
	// b,c,h,w
	std::vector<int64_t> input_node_dims{1,1,512,512};
	size_t input_tensor_size;

	std::vector<const char*> out_node_names;
	size_t out_tensor_size;

	int image_h;
	int image_w;
	int refsize=512;

	vector<vector<float>> normalized_param{ {0.485, 0.456, 0.406},{0.229, 0.224, 0.225} };
	cv::Mat normalize(cv::Mat &image);
	vector<cv::Mat> preprocess(cv::Mat &image);
public:

	HumanSeg() =delete;
	HumanSeg(std::wstring model_path, int num_threads);
	
	cv::Mat predict(cv::Mat &image,const string image_or_video);
};
