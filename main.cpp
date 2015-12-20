#include <cstdlib>
#include <iostream>

#include <opencv2/opencv.hpp>

void Run(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cout << "usage: DisplayImage.out <Image_Path>" << std::endl;
		return;
	}

	cv::Mat image;
	image = cv::imread(argv[1], 1);

	if (!image.data)
	{
		std::cout << "No image data" << std::endl;
		return;
	}

	cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display Image", image);
	cv::waitKey(0);
}

int main(int argc, char **argv)
{
	std::cout << "Starting..." << std::endl;

	try {
		Run(argc, argv);
	} catch (const std::exception &exc) {
		std::cerr << exc.what();
		return EXIT_FAILURE;
	} catch (...) {
		std::cerr << "Unhandled exception!";
		return EXIT_FAILURE;
	}

	std::cout << "Exiting successfully!" << std::endl;

	return EXIT_SUCCESS;
}
