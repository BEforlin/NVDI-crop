#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include "dirent.h"
#include <unistd.h>
#include <string>
#define GetCurrentDir getcwd

/*
Cenario assumptions:
  - Drone moves in a single known direction;
  - We know the image order from the drone feed;
  - We have set drone movement direction to south;
  - There is no variation in the number of ground markers in a line;
  - There is no vizual pollution (trucks, cows, etc) that can be confused with a marker

Quadrant division:

      |1|2|
      |4|3|

Use vconcat(mat1, mat2, dst) and hconcat(mat1, mat2, dst)!!!

Crop image opencv:

Mat image=imread("image.png",1);

int startX=200,startY=200,width=100,height=100

Mat ROI(image, Rect(startX,startY,width,height));

Mat croppedImage;

// Copy the data into new matrix
ROI.copyTo(croppedImage);

imwrite("newImage.png",croppedImage);




*/

//--------------namespaces--------------
using namespace cv;
using namespace std;

//--------------headers--------------
void processing_calback(int, void*);
void capture_im(string fname, int Height, int Width);
void quadrant_find(int im_act);
int dir_stuf(char *argv[]);
void processing_hack(void);
// void center_mass(void);
void ndvi_colormap(Mat bgr_image, Mat nir_image);
Mat dst;

//--------------globals--------------
Mat dilate_element = getStructuringElement(MORPH_RECT, Size(2, 2));
Mat src;
Mat bottom_im;
Mat top_im;
Mat top_im_nir;
Mat bottom_im_nir;
Mat AGORAFOICARALHO(Size(1500, 2500), CV_8UC3);;

vector<vector<vector<Point> > >contours_list;
vector<string> file_list = vector<string>();
int contours_it = 0;
vector<vector<Point2f> > mc_out;

vector<Point2f> listing_bot;//os contours em ordem
vector<Point2f> listing_top;//os contours em ordem

std::vector<Point2f> listing_1;
std::vector<Point2f> listing_2;
std::vector<Point2f> listing_3;

int total_y_size = 2500;
Mat finalIm(Size(1500, total_y_size), CV_8UC3);
Mat finalIm_nir(Size(1500, total_y_size), CV_8UC1);

char cCurrentPath[FILENAME_MAX];

string source_window = "Main";
string source_window_nir  = "nir";
string source_window_final  = "final";

int main(int argc, char** argv){
  //--------------GUI operations--------------
  //namedWindow( canny_window , WINDOW_NORMAL);
  namedWindow( source_window , WINDOW_NORMAL);
  namedWindow( source_window_nir , WINDOW_NORMAL);
  namedWindow( source_window_final , WINDOW_NORMAL);

  //--------------file aquisition--------------
  const char* filename = argc >= 2 ? argv[1] : "\0";
  if (argc == 1) {
    cout << "No input arguments! Need folder name (folder must be in dir)!" << endl;
    return -1;
  }

  if (dir_stuf(argv) < 0) {
    return -1;
  }

  //--------------file iteration--------------
  sort (file_list.begin(), file_list.end());

  Mat im_stiched;

  vector<Mat> im_vector;

  for (int i = 0;i < 3;i++) {
    capture_im(file_list[i], 0, 0);

    processing_hack();

    im_vector.push_back(src);

    // imshow(file_list[i], im_vector[i]);
  }

  for (int i = 3; i < 6; i++) {

    capture_im(file_list[i], 1, 1);

    im_vector.push_back(src);
  }
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    int i = 1;

    quadrant_find(i);

    top_im = im_vector[i];
    bottom_im = im_vector[i-1];
    top_im_nir = im_vector[i+3];
    bottom_im_nir = im_vector[i+2];

    vector<Point2f> borders_top;
    vector<Point2f> borders_bottom;

    borders_top.push_back(Point2f(0,0));
    borders_top.push_back(Point2f(0,top_im.rows));
    borders_top.push_back(Point2f(top_im.cols,0));
    borders_top.push_back(Point2f(top_im.cols,top_im.rows));

    borders_bottom.push_back(Point2f(0,0));
    borders_bottom.push_back(Point2f(0,bottom_im.rows));
    borders_bottom.push_back(Point2f(bottom_im.cols,0));
    borders_bottom.push_back(Point2f(bottom_im.cols,bottom_im.rows));

    Mat homography = findHomography(listing_top, listing_bot, RANSAC);
    Mat warped_im_1;
    Mat warped_im_1_nir;

    perspectiveTransform(borders_top, borders_bottom, homography);

    int cutoff_x = 0;
    int cutoff_y = 0;

    for (int i = 0; i < borders_bottom.size(); i++) {
      // std::cout << "boadad " << borders_bottom[i] << '\n';
      if (borders_bottom[i].x < cutoff_x) {
        cutoff_x = borders_bottom[i].x;
      }
      if (borders_bottom[i].y < cutoff_y) {
        cutoff_y = borders_bottom[i].y;
      }
    }

    // total_y_size += -cutoff_y +bottom_im.rows;

    // Mat roi1 = Mat(finalIm, Rect(0,0,top_im.cols,-cutoff_y));
    // Mat roi1 = Mat(finalIm, Rect(-100,-100,123,1123));

    // Mat roi2 = Mat(finalIm, Rect(-borders_bottom[0].x,-cutoff_y,bottom_im.cols,bottom_im.rows));

    int x_size_wtf = -borders_bottom[0].x;
    int y_size_wtf = top_im.rows-cutoff_y;

    cutoff_x = 0;
    Mat A = Mat::eye(3,3,CV_64F);
    A.at<double>(0,2)= -cutoff_x;
    A.at<double>(1,2)= -cutoff_y;
    Mat F = A*homography;

    warpPerspective(bottom_im, warped_im_1, F, Size(bottom_im.cols, bottom_im.rows), INTER_CUBIC);
    warpPerspective(bottom_im_nir, warped_im_1_nir, F, Size(bottom_im.cols, bottom_im.rows), INTER_CUBIC);

    Mat croppedImage_1 = warped_im_1(Rect(0, 0, top_im.cols, -cutoff_y));
    Mat croppedImage_1_nir = warped_im_1_nir(Rect(0, 0, top_im.cols, -cutoff_y));

    // Mat finalIm;
    // Mat finalIm_nir;

    vconcat(croppedImage_1, top_im, finalIm);
    vconcat(croppedImage_1_nir, top_im_nir, finalIm_nir);

    // std::cout << "nir" << << '\n';

    // warped_im_1.copyTo(roi1);
    // imshow("roi1", roi1);
// imshow("cara  ", warped_im_1);
    // top_im.copyTo(roi2);
    // imshow("roi2", roi2);

    listing_bot.clear();
    listing_top.clear();
    borders_top.clear();
    borders_bottom.clear();

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


    i = 2;

    quadrant_find(i);

    top_im = im_vector[i];
    bottom_im = im_vector[i-1];
    top_im_nir = im_vector[i+3];
    bottom_im_nir = im_vector[i+2];

    borders_top.push_back(Point2f(0,0));
    borders_top.push_back(Point2f(0,top_im.rows));
    borders_top.push_back(Point2f(top_im.cols,0));
    borders_top.push_back(Point2f(top_im.cols,top_im.rows));

    borders_bottom.push_back(Point2f(0,0));
    borders_bottom.push_back(Point2f(0,bottom_im.rows));
    borders_bottom.push_back(Point2f(bottom_im.cols,0));
    borders_bottom.push_back(Point2f(bottom_im.cols,bottom_im.rows));

    homography = findHomography(listing_bot, listing_top, RANSAC);

    perspectiveTransform(borders_top, borders_bottom, homography);

    cutoff_x = 0;
    cutoff_y = 0;

    for (int i = 0; i < borders_bottom.size(); i++) {
      // std::cout << "boadad " << borders_bottom[i] << '\n';
      if (borders_bottom[i].x > cutoff_x) {
        cutoff_x = borders_bottom[i].x;
      }
      if (borders_bottom[i].y > cutoff_y) {
        cutoff_y = borders_bottom[i].y;
      }
    }

    cutoff_x = 0;
    A = Mat::eye(3,3,CV_64F);
    A.at<double>(0,2)= 0;
    A.at<double>(1,2)= -top_im.rows;
    F = A*homography;

    Mat warped_im;
    Mat warped_im_nir;

    // double fullBorder = borders_top[3].y - borders_bottom[0].y - borders_bottom[3].y;
    double fullBorder = borders_top[3].y - borders_bottom[3].y;

    warpPerspective(top_im, warped_im, F, Size(bottom_im.cols, bottom_im.rows), INTER_CUBIC);
    warpPerspective(top_im_nir, warped_im_nir, F, Size(bottom_im.cols, bottom_im.rows), INTER_CUBIC);

    // imshow("waradads", warped_im);

    Mat croppedImage = warped_im(Rect(0 , 0 , top_im.cols ,-fullBorder));
    Mat croppedImage_nir = warped_im_nir(Rect(0 , 0 , top_im.cols ,-fullBorder));

    // imshow("reste", croppedImage);

    //std::cout << x_size_wtf+borders_bottom[0].x << "  "<< cutoff_y+top_im.rows << "  "<<top_im.cols << "  "<<fullBorder << '\n';
    // Mat roi3 = Mat(finalIm, Rect(x_size_wtf+borders_bottom[0].x , cutoff_y+top_im.rows , top_im.cols ,-fullBorder));

    // warped_im.copyTo(roi3);

    vconcat(finalIm, croppedImage, finalIm);
    vconcat(finalIm_nir, croppedImage_nir, finalIm_nir);

    imshow(source_window, finalIm);
    imshow(source_window_nir, finalIm_nir);

    // std::cout << "nir: " << finalIm_nir.size() << " ada " << finalIm.size()<< '\n';

    imwrite("nir.bmp", finalIm_nir);
    imwrite("normal.bmp", finalIm);
    ndvi_colormap(finalIm,finalIm_nir);

    imshow(source_window_final, dst);


  waitKey(0);

  return 0;
}

//--------------library--------------
void quadrant_find(int im_act){
  int rows = src.rows;
  int cols = src.cols;
  // std::cout << "im_act"<< im_act << '\n';
  for (int i = im_act; i >= (im_act-1) ; --i) {//itera lista de contornos
  // cout << "\ni>" << i << "\n";
    for (int j = 0; j < contours_list.at(i).size(); j++) {//itera por cada contorno
      //cout << "\nj>" << j << contours_list.at(i).at(j).at(0).y << "\n";
      if ((contours_list.at(i).at(j).at(0).y < rows/2) && (contours_list.at(i).at(j).at(0).x < cols/2 )) {//1 quad
        if (i == (im_act)  ) {
          // std::cout << "4: " << i <<'\n';
          for (int k = 0; k < 6; k++) {
            listing_bot.insert(listing_bot.end(), contours_list.at(i).at(j).at(k));
          }
        }
      }else if ((contours_list.at(i).at(j).at(0).y < rows/2) && (contours_list.at(i).at(j).at(0).x > cols/2 )) {//2 quad
        if (i == (im_act)  ) {
          // std::cout << "3: "<< i << '\n';
          for (int k = 0; k < 6; k++) {
            listing_bot.insert(listing_bot.end(), contours_list.at(i).at(j).at(k));
          }
        }
      }else if ((contours_list.at(i).at(j).at(0).y > rows/2) && (contours_list.at(i).at(j).at(0).x > cols/2 )) {//3 quad
        if (i == (im_act-1)  ) {
          // std::cout << "2: "<< i << '\n';
          for (int k = 0; k < 6; k++) {
            listing_top.insert(listing_top.end(), contours_list.at(i).at(j).at(k));
          }
        }
      }else if ((contours_list.at(i).at(j).at(0).y > rows/2) && (contours_list.at(i).at(j).at(0).x < cols/2 )) {//4 quad
        if (i == (im_act-1)  ) {
          // std::cout << "1: "<< i << '\n';
          for (int k = 0; k < 6; k++) {
            listing_top.insert(listing_top.end(), contours_list.at(i).at(j).at(k));
          }
        }
      }
    }
  }
}

void processing_hack(void){
  Mat dst_hsv, dst_black, dst_white;
  Mat color_filtered = src.clone();
  dst_hsv = src.clone();

  vector<vector<Point> > contours;
  vector<vector<Point> > contours_pre;

  int white_high = 255, white_low = 242, black_high= 75, black_low= 0;

  cvtColor(src, dst_hsv, COLOR_BGR2HSV);

  //Color Filter
  // for black
  cv::inRange(dst_hsv, cv::Scalar(0, 0, black_low, 0), cv::Scalar(180, 255, black_high, 0), dst_black);

  // for white
  cv::inRange(dst_hsv, cv::Scalar(0, 0, white_low, 0), cv::Scalar(180, 255, white_high, 0), dst_white);

  color_filtered = dst_black + dst_white;

  // imshow("oi", color_filtered);

  //Blur the image to get rid of high frequency noise
  medianBlur(color_filtered, color_filtered, 3);

  // imshow("super", color_filtered);

  //Erode actually dilates because of invert
  erode(color_filtered, color_filtered, dilate_element, Point(-1,-1), 2);

  // imshow("superasas", color_filtered);

  //Find contours of the remaining objects
  findContours( color_filtered, contours_pre, -1, 2, Point(-1,-1));

  //Eliminates image border by area and momentun
  for (int i = 0; i< contours_pre.size(); i++){
      double area = contourArea(contours_pre[i]);
      if ((area > 10) && (area < 1000)){
        contours.push_back(contours_pre[i]);
      }
  }
  contours_list.push_back(contours);

  //Draws the actual borders of the objects
  drawContours( src, contours, -1, Scalar(0,0,255), 2);

  //imshow("source_window", src);
}

void capture_im(string fname, int Height, int Width){
  if (Height == 1) {
    char cCurrentPath_2[FILENAME_MAX];

    strcpy(cCurrentPath_2, cCurrentPath);
    strcat(cCurrentPath_2,fname.c_str());
    src = imread(cCurrentPath_2, IMREAD_GRAYSCALE );

    if(src.empty()){
      cout << "Error reading source file! " << fname << endl;
    }
  }else{
    char cCurrentPath_2[FILENAME_MAX];

    strcpy(cCurrentPath_2, cCurrentPath);
    strcat(cCurrentPath_2,fname.c_str());
    src = imread(cCurrentPath_2);

    if(src.empty()){
      cout << "Error reading source file! " << fname << endl;
    }
  }
}

int dir_stuf(char *argv[]){
  DIR *dir;
  struct dirent *ent;


  GetCurrentDir(cCurrentPath, sizeof(cCurrentPath));

  strcat(cCurrentPath,"/");
  strcat(cCurrentPath,argv[1]);
  strcat(cCurrentPath,"/");

  if ((dir = opendir (cCurrentPath)) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      if (strcmp(ent->d_name, ".") == 0)  {
        continue;
      } if (strcmp(ent->d_name, "..") == 0)  {
        continue;
      }
      file_list.push_back(string(ent->d_name));
    }
    closedir (dir);
  } else {
    cout << "Dir does not exist!";
    return -1;
  }
}

void processing_calback(int, void*){
  //debugging
}

void ndvi_colormap(cv::Mat bgr_image, cv::Mat nir_image){
  //--------------NDVI-----------------------------//
  Mat red_image(bgr_image.size(),nir_image.type());
  Mat sum, subtraction, ndvi, ndvi_scaled, ndvi_clrmap,trans;
  //cvtColor(nir_image, nir_image, COLOR_BGR2GRAY);
  // criacao dos canais para dar o split na imagem
  vector<Mat> channels;

  /*//ignorar, as imagens serao passadas como parametros
  nir_image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  bgr_image = imread(argv[2], CV_LOAD_IMAGE_COLOR);
  //--------------------------------------------------*/

  subtraction = Mat::ones(nir_image.rows, nir_image.cols, CV_32FC1);
  sum = Mat::ones(nir_image.rows, nir_image.cols, CV_32FC1);
  ndvi = Mat::ones(nir_image.rows, nir_image.cols, CV_32FC1);

  split(bgr_image, channels);
  red_image = channels[2];
  //
  // align_images(red_image, nir_image, nir_image);
  //
  // //Converte para float pra nao travar os valores em 8 bits...
  red_image.convertTo(red_image, CV_32FC1);
  nir_image.convertTo(nir_image, CV_32FC1);
  //
  // // Calcula o NDVI
  subtract(nir_image, red_image, subtraction);
  add(nir_image, red_image, sum);
  divide(subtraction, sum, ndvi);

  // converte novamente para uint8
  ndvi.convertTo(ndvi_scaled, CV_8UC1, 255.0);

  equalizeHist(ndvi_scaled,ndvi_scaled);

  // aplica um colormap
  applyColorMap(ndvi_scaled, ndvi_clrmap, COLORMAP_JET);

  dst = ndvi_clrmap;
  // imshow("NDVI Colormapped", ndvi_clrmap);
}
