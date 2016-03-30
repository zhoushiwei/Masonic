//
//  main.cpp
//  Masoic
//
//  Created by zhoushiwei on 16/3/13.
//  Copyright © 2016年 zhoushiwei. All rights reserved.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>

/**
 * @brief Computes mosaic image from input picture.
 *
 * @param source [in]   Source image. Should be of CV_8UC3 or CV_8UC4 type.
 * @param gridSize [in] Desired grid size. The smaller grid size, the more detail mosaic will be.
 */
cv::Mat mosaic(cv::Mat source, const int gridSize = 24);

int main(int argc, const char * argv[])
{
    const cv::String keys =
    "{help h usage ? |      | print this message    }"
    "{@source        |      | source image          }"
    "{@output        |      | destination filename  }"
    "{gridSize       |32    | mosaic fadet size     }"
    ;
    
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV Mosaicing demo v1.0.0");
    
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    cv::String src = "/Users/zhoushiwei/百度云同步盘/明星/陈乔恩.jpg";
    cv::String dst = "/Users/zhoushiwei/Dropbox/Masoic/Masoic/img.jpg";
    const int gridSize = parser.get<int>("gridSize");
    
    if (src.empty()) {
        std::cerr << "Source image is not specified" << std::endl;
        parser.printMessage();
        return 1;
    }
    
    if (dst.empty()) {
        std::cerr << "Output image is not specified" << std::endl;
        parser.printMessage();
        return 1;
    }
    
    if (gridSize < 0) {
        std::cerr << "Grid size cannot be less than 1" << std::endl;
        parser.printMessage();
        return 1;
    }
    
    cv::Mat source = cv::imread(src);
    cv::Mat m = mosaic(source, gridSize);
    cv::imwrite(dst, m);
    cv::imshow("origin",source);
    cv::imshow("dst",m);
    cv::waitKey();
    return 0;
}

void computeTriangleGrid(
                         std::vector<cv::KeyPoint>& grid,
                         cv::Rect frame,
                         int gridSize,
                         int pointOffset)
{
    grid.clear();
    const int w = std::ceil(frame.width / (float)gridSize);
    const int h = std::ceil(frame.height / (float)gridSize);
    
    cv::RNG rnd;
    
    // Add four seed points starting at image corners
    grid.push_back(cv::KeyPoint(0, 0, 0));
    grid.push_back(cv::KeyPoint(frame.width-1, 0, 0));
    grid.push_back(cv::KeyPoint(0, frame.height-1, 0));
    grid.push_back(cv::KeyPoint(frame.width-1, frame.height-1, 0));
    
    for (int i = 0; i <= w; i++)
    {
        for (int j = 0; j <= h; j++)
        {
            int x = i * gridSize + (j % 2 ? gridSize/2 : 0);
            int y = j * gridSize;
            
            if (pointOffset > 0 && (i > 0 && j > 0 && i < w && j < h))
            {
                x += rnd.uniform(-pointOffset, pointOffset);
                y += rnd.uniform(-pointOffset, pointOffset);
            }
            
            cv::Point pt(std::max(0,std::min(frame.width-1,x)),
                         std::max(0,std::min(frame.height-1,y)));
            
            grid.emplace_back(pt, 0);
        }
    }
}

class ParallelMosaicBody : public cv::ParallelLoopBody
{
    cv::Mat *                      _img;
    const std::vector<cv::Vec6f> * _triangleList;
public:
    ParallelMosaicBody(cv::Mat& img, const std::vector<cv::Vec6f>& triangleList)
    : _img(&img)
    , _triangleList(&triangleList)
    {
    }
    
    void updateWithAverageColor(cv::Mat& img, const cv::Vec6f& t) const
    {
        std::vector<cv::Point> pt = {
            cv::Point(t[0], t[1]),
            cv::Point(t[2], t[3]),
            cv::Point(t[4], t[5])
        };
        
        cv::Rect r = cv::boundingRect(pt);
        if (r.x < 0 || r.y < 0
            || (r.x + r.width) > img.cols
            || (r.y + r.height) > img.rows)
            return;
        
        cv::Mat_<uint8_t> mask(r.height, r.width);
        
        for (int j = r.x; j < r.x + r.width; j++)
            for (int i = r.y; i < r.y + r.height; i++)
                if (cv::pointPolygonTest(pt, cv::Point(j,i), false) >= 0)
                    mask(i - r.y, j - r.x) = 1;
                else
                    mask(i - r.y, j - r.x) = 0;
        
        cv::Scalar avg = cv::mean(img(r), mask);
        img(r).setTo(avg, mask);
    }
    
    void operator()(const cv::Range &r) const override
    {
        auto& img = *_img;
        const auto& triangleList = *_triangleList;
        
        for (int i = r.start; i < r.end; i++)
        {
            auto& facet = triangleList[i];
            updateWithAverageColor(img, facet);
        }
    }
};

cv::Mat mosaic(cv::Mat source, const int gridSize)
{
    const cv::Rect frame(0,0, source.cols, source.rows);
    
    cv::Mat gray;
    cv::cvtColor(source, gray, cv::COLOR_RGB2GRAY);
    auto detector = cv::GFTTDetector::create(1024, 0.05, gridSize/2);
    
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray, keypoints);
    
    // Remove near-border keypoints
    cv::KeyPointsFilter::runByImageBorder(
                                          keypoints,
                                          frame.size(),
                                          gridSize/3);
    
    // Create regular grid for better mosaicing
    std::vector<cv::KeyPoint> grid;
    computeTriangleGrid(grid, frame, gridSize, gridSize/2);
    
    // Insert detected keypoints to employ image structure
    for (auto kp: keypoints) {
        // Convert keypoints to integer
        cv::Point pt = kp.pt;
        grid.emplace_back(pt,0);
    }
    
    // Remove duplicate keypoints to avoid appear
    // of duplicate faces in subdivision
    cv::KeyPointsFilter::removeDuplicated(grid);
    
    cv::Subdiv2D subdiv(frame);
    for (auto keypoint: grid)
        subdiv.insert(keypoint.pt);
    
    // Convert image to floating point to reduce preicision loss during averaging
    cv::Mat sourcef;
    source.convertTo(sourcef, CV_32F);
    
    // Fill mosaic
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    
    cv::parallel_for_(cv::Range(0, triangleList.size()), 
                      ParallelMosaicBody(sourcef, triangleList), 32);
    
    // Convert image to back to byte range
    cv::Mat mosaic;
    sourcef.convertTo(mosaic, CV_8U);
    return mosaic;
}