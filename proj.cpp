#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define MAX_FRAMES 521


// Global state variables

bool is_mouse_down = false, is_mouse_up = false;
Mat frame, processed_frame;
Rect selection;
Point origin, end_points;
vector<Rect> selections;
Mat roi;

// Mouse callback event
void MouseCallbackEvent(int event, int x, int y, int flags, void* userdata)
{

    printf("x %d y %d  event %d  flags %d\n", x, y, event, flags);

    if (x >= frame.cols) {
        x = frame.cols - 1;
    }
    else if (x < 0) {
        x = 0;
    }

    if (y >= frame.rows) {
        y = frame.rows - 1;
    }
    else if (y < 0) {
        y = 0;
    }

    if (event == EVENT_LBUTTONDOWN)
    {

        is_mouse_down = true;
        origin = Point(x, y);
        printf("mouse down!\n");
    }

    if (event == EVENT_LBUTTONUP)
    {
        is_mouse_up = true;
        end_points = Point(x, y);
        selections.push_back(selection);
        printf("mouse up!\n");

    }
    if (is_mouse_down == true && is_mouse_up == false) {

        Point xy_point;
        xy_point = Point(x, y);

        rectangle(processed_frame, origin, xy_point, Scalar(0, 0, 255)); //drawing a rectangle continuously
        imshow("Output", processed_frame);

    }

    if (is_mouse_down == true && is_mouse_up == true)
    {
        selections.erase(selections.end() - 1);

        selection.x = MIN(end_points.x, origin.x);
        selection.y = MIN(end_points.y, origin.y);
        selection.width = std::abs(end_points.x - origin.x) + 1;
        selection.height = std::abs(end_points.y - origin.y) + 1;

        if (selection.width > 0 && selection.height > 0)
        {
            Mat crop(frame, selection); //Selecting a ROI(region of interest) from the original pic
            roi = crop.clone();
            imshow("Template", roi);
            is_mouse_down = false;
            is_mouse_up = false;
        }
    }
}


char path[4096];


Ptr<ORB> p_orb;
std::vector<cv::KeyPoint> keypoints_roi, keypoints_img;
Mat descriptor1, descriptor2;
const float GOOD_MATCH_PERCENT = 0.75f;

void orbFeatureDetector(Mat frame, Mat roi) {

    printf("img1 rows %d cols %d data %p\n", frame.rows, frame.cols, frame.data);
    printf("img2 rows %d cols %d data %p\n", roi.rows, roi.cols, roi.data);

    // Make a visualization image
    Mat vis1 = frame.clone();
    Mat vis2 = roi.clone();

    // Calculate the ORB feature points of both images
    p_orb = ORB::create(30000);
    p_orb->detect(frame, keypoints_img);
    p_orb->compute(frame, keypoints_img, descriptor1);
    p_orb->detect(roi, keypoints_roi);
    p_orb->compute(roi, keypoints_roi, descriptor2);

    // Plot the feature points
    for (int i = 0; i < keypoints_img.size(); i++)
        circle(vis1, Point(keypoints_img[i].pt), 1, Scalar(255, 0, 0), 3);

    for (int i = 0; i < keypoints_roi.size(); i++)
        circle(vis2, Point(keypoints_roi[i].pt), 1, Scalar(255, 0, 0), 3);
    processed_frame = vis1.clone();

    // Match the keypoints
    std::vector< DMatch > matches;
    //	std::vector< std::vector<DMatch> > kmatches;
    Ptr<BFMatcher> bf = BFMatcher::create(NORM_HAMMING);
    bf->match(descriptor2, descriptor1, matches);


    std::vector<Point2f> roi_vec, image_vec;
    for (size_t i = 0; i < matches.size(); i++)

    {
        //-- Get the keypoints from the matches
        roi_vec.push_back(keypoints_roi[matches[i].queryIdx].pt);
        image_vec.push_back(keypoints_img[matches[i].trainIdx].pt);
    }

    Mat H;
    if ((!roi_vec.empty()) && (!image_vec.empty())) {
        H = findHomography(roi_vec, image_vec, RANSAC);
    }

    //-- Get the corners from the roi ( the object to be "detected" )
    std::vector<Point2f> roi_corners(4);
    roi_corners[0] = Point2f(0, 0);
    roi_corners[1] = Point2f((float)roi.cols, 0);
    roi_corners[2] = Point2f((float)roi.cols, (float)roi.rows);
    roi_corners[3] = Point2f(0, (float)roi.rows);

    std::vector<Point2f> image_corners(4);
    if (!H.empty())
    {
        perspectiveTransform(roi_corners, image_corners, H);
    }

    // draw the lines around the features 
    line(vis1, image_corners[0], image_corners[1], Scalar(0, 255, 0), 4);

    line(vis1, image_corners[1], image_corners[2], Scalar(0, 255, 0), 4);

    line(vis1, image_corners[2], image_corners[3], Scalar(0, 255, 0), 4);

    line(vis1, image_corners[3], image_corners[0], Scalar(0, 255, 0), 4);

    // Show the images
    imshow("Output", vis1);
    waitKey(10);

    imshow("Template", vis2);
    waitKey(10);

}

int main()
{

    namedWindow("Output", 1);

    // Set the mouse callback
    setMouseCallback("Output", MouseCallbackEvent, NULL);

    //----
    // Read the video as a series of frames
    //----
    int frameno = 0;
    while (1)
    {

        sprintf(path, "pictureframes/frame%05d.jpg", frameno);
        frame = imread(path);

        if (frame.data == 0) {
            printf("ERROR: could not read frame %s\n", path);
            break;
        }



        if (!roi.empty()) {
            orbFeatureDetector(frame, roi);
        }
        else {
            processed_frame = frame.clone();
            imshow("Output", frame);
            waitKey(10);
        }

        waitKey(30);
        // Next frame, and loop around
        frameno = (frameno + 1) % MAX_FRAMES;
    }

    return 0;
}