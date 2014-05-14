#include"stdafx.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "glut.h"      
#include "GL/glu.h"    
#include "GL/gl.h"   
#include <math.h>
#include <time.h>

using namespace std;     
using namespace cv;

const float zNear = 0.05;			
const float zFar = 500.0;
int width,height;
int draw=0;
Point FIX_X(0,0),FIX_Y(0,0),FIX_Z(0,0);
float skew_x,skew_y,skew_z;
VideoCapture cap(0);
Mat tmp,test;
Mat intrinsic_Matrix(3,3, CV_64F);
Mat distortion_coeffs(8,1, CV_64F);
Mat Projection(4,4, CV_64FC1);
double largest_area;
int largest_contour_index;
int n=0; 
int no_of_fingers=0;
vector<vector<pair<float,Point>>> position;
vector<int> finger_count;
Point first,second,third;
float size_of_pot=10;
int rot_angle=10;

float distanceP2P(Point a, Point b){
	float d= sqrt(fabs( pow(a.x-b.x,2) + pow(a.y-b.y,2) )) ;  
	return d;
}
float getAngle(Point s, Point f, Point e){
	float l1 = distanceP2P(f,s);
	float l2 = distanceP2P(f,e);
	float dot=(s.x-f.x)*(e.x-f.x) + (s.y-f.y)*(e.y-f.y);
	float angle = acos(dot/(l1*l2));
	angle=angle*180/3.147;
	return angle;
}


String intToString(int number){
	stringstream ss;
	ss << number;
	string str = ss.str();
	return str;
}

bool pairCompare(const pair<float,Point>&i, const pair<float,Point>&j) {
	return i.first <j.first;

}

GLfloat* convertMatrixType(const cv::Mat& m)
{
	typedef double precision;

	Size s = m.size();
	GLfloat* mGL = new GLfloat[s.width*s.height];

	for(int ix = 0; ix < s.width; ix++)
	{
		for(int iy = 0; iy < s.height; iy++)
		{
			mGL[ix*s.height + iy] = m.at<precision>(iy, ix);
		}
	}

	return mGL;
}

void generateProjectionModelview(const cv::Mat& calibration, const cv::Mat& rotation, const cv::Mat& translation, cv::Mat& projection, cv::Mat& modelview)
{
	typedef double precision;

	projection.at<precision>(0,0) = 2*calibration.at<precision>(0,0)/width;
	projection.at<precision>(1,0) = 0;
	projection.at<precision>(2,0) = 0;
	projection.at<precision>(3,0) = 0;

	projection.at<precision>(0,1) = 0;
	projection.at<precision>(1,1) = 2*calibration.at<precision>(1,1)/height;
	projection.at<precision>(2,1) = 0;
	projection.at<precision>(3,1) = 0;

	projection.at<precision>(0,2) = 1-2*calibration.at<precision>(0,2)/width;
	projection.at<precision>(1,2) = -1+(2*calibration.at<precision>(1,2)+2)/height;
	projection.at<precision>(2,2) = (zNear+zFar)/(zNear - zFar);
	projection.at<precision>(3,2) = -1;

	projection.at<precision>(0,3) = 0;
	projection.at<precision>(1,3) = 0;
	projection.at<precision>(2,3) = 2*zNear*zFar/(zNear - zFar);
	projection.at<precision>(3,3) = 0;


	modelview.at<precision>(0,0) = rotation.at<precision>(0,0);
	modelview.at<precision>(1,0) = rotation.at<precision>(1,0);
	modelview.at<precision>(2,0) = rotation.at<precision>(2,0);
	modelview.at<precision>(3,0) = 0;

	modelview.at<precision>(0,1) = rotation.at<precision>(0,1);
	modelview.at<precision>(1,1) = rotation.at<precision>(1,1);
	modelview.at<precision>(2,1) = rotation.at<precision>(2,1);
	modelview.at<precision>(3,1) = 0;

	modelview.at<precision>(0,2) = rotation.at<precision>(0,2);
	modelview.at<precision>(1,2) = rotation.at<precision>(1,2);
	modelview.at<precision>(2,2) = rotation.at<precision>(2,2);
	modelview.at<precision>(3,2) = 0;

	modelview.at<precision>(0,3) = translation.at<precision>(0,0);
	modelview.at<precision>(1,3) = translation.at<precision>(1,0);
	modelview.at<precision>(2,3) = translation.at<precision>(2,0);
	modelview.at<precision>(3,3) = 1;

	// This matrix corresponds to the change of coordinate systems.
	static double changeCoordArray[4][4] = {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};
	static Mat changeCoord(4, 4, CV_64FC1, changeCoordArray);

	modelview = changeCoord*modelview;
}


void calibrate(Mat &intrinsic_Matrix,Mat &distortion_coeffs)
{

	vector< vector< Point2f> > AllimagePoints;
	vector< vector< Point3f> > AllobjectPoints;
	char str[100];
	stringstream st;
	int no_of_images=1;
	Size imagesize;
	Mat gray;
	while(no_of_images<=14)
	{
		st<<"E:/SelectedImages/Selected"<<++no_of_images<<".jpg";
		String strcopy3=st.str();
		st.str("");
		Mat img=imread(strcopy3,1);
		if(!img.data)
			break;
		imagesize=Size(img.rows,img.cols);
		cvtColor(img, gray, CV_RGB2GRAY);
		vector< Point2f> corners;  
		bool sCorner =false;
		sCorner=findChessboardCorners(gray, Size(7,7), corners);
		if(sCorner)
		{

			cornerSubPix(gray, corners, Size(11,11), Size(-1,-1), TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(img, Size(7,7), corners, sCorner);
			if(corners.size() == 7*7)
			{
				vector< Point2f> v_tImgPT;
				vector< Point3f> v_tObjPT;
				for(int j=0; j< corners.size(); ++j)
				{
					Point2f tImgPT;
					Point3f tObjPT;

					tImgPT.x = corners[j].x;
					tImgPT.y = corners[j].y;

					tObjPT.x = j%7*3;
					tObjPT.y = j/7*3;
					tObjPT.z = 0;

					v_tImgPT.push_back(tImgPT);
					v_tObjPT.push_back(tObjPT);     
				}
				AllimagePoints.push_back(v_tImgPT);
				AllobjectPoints.push_back(v_tObjPT);
			}

		}
		st<<"E:/DetectedImages/Detected"<<no_of_images+1<<".jpg";
		String strcopy1=st.str();
		st.str("");
		imwrite(strcopy1,img);
		//imshow("pattern",img);
		//cvWaitKey(30);
	}
	vector< Mat> rvecs, tvecs;
	if(AllimagePoints.size()>0)
	{
		calibrateCamera(AllobjectPoints,AllimagePoints,imagesize, intrinsic_Matrix, distortion_coeffs, rvecs, tvecs);
	}

}


void renderBackgroundGL(const cv::Mat& image)
{
	
	GLint polygonMode[2];
	glGetIntegerv(GL_POLYGON_MODE, polygonMode);
	glPolygonMode(GL_FRONT, GL_FILL);
	glPolygonMode(GL_BACK, GL_FILL);

	
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	
	static bool textureGenerated = false;
	static GLuint textureId;
	if (!textureGenerated)
	{
		glGenTextures(1, &textureId);

		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		textureGenerated = true;
	}

	// Copy the image to the texture.
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.size().width, image.size().height, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, image.data);

	// Draw the image.
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_TRIANGLES);
	glNormal3f(0.0, 0.0, 1.0);

	glTexCoord2f(0.0, 1.0);
	glVertex3f(0.0, 0.0, 0.0);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(0.0, 1.0, 0.0);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(1.0, 0.0, 0.0);

	glTexCoord2f(1.0, 1.0);
	glVertex3f(1.0, 0.0, 0.0);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(0.0, 1.0, 0.0);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(1.0, 1.0, 0.0);
	glEnd();
	glDisable(GL_TEXTURE_2D);

	// Clear the depth buffer so the texture forms the background.
	glClear(GL_DEPTH_BUFFER_BIT);

	// Restore the polygon mode state.
	glPolygonMode(GL_FRONT, polygonMode[0]);
	glPolygonMode(GL_BACK, polygonMode[1]);
}




void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
	namedWindow("live",1);
	Mat gray1,test,modelview,dis_img,thresh,img1;

	Mat rvec(3,1,DataType<double>::type);
	Mat tvec(3,1,DataType<double>::type);

	modelview.create(4, 4, CV_64FC1);
	//Projection.create(4, 4, CV_64FC1);

	vector< Point2f> corners1;  
	vector< Point2f> imagePoints1;
	vector< Point3f> objectPoints1;
	largest_area=0;
	largest_contour_index=0;

	clock_t clock_1 = clock();
	cap >> dis_img;
	//resize(dis_img,dis_img,Size(180,180),0,0);
	if(!dis_img.data)
	{
		exit(3);
	}
	img1=dis_img.clone();
	dis_img.copyTo(img1);	
	//resize(img1,img1,Size(180,180),0,0);
	cvtColor(dis_img,dis_img,COLOR_BGR2YCrCb);
	inRange(dis_img,Scalar(0,133,77),Scalar(255,173,127),thresh);
	clock_t clock_2 = clock();
	cout<<"threshold(Skin Color Segmentation) time is :"<<(double)(clock_2-clock_1)<<endl;
	dilate(thresh,thresh,Mat());
	blur(thresh,thresh,Size(5,5),Point(-1,-1),BORDER_DEFAULT);
	vector<vector<Point>> contours;
	vector<Point> FingerTips;
	vector<Vec4i> hierachy; 
	vector<Vec4i> defects;
	vector<Point> defect_circle;
	vector<vector<Point>> hull(1);
	Point2f  center;
	float radius;
	clock_t clock_3 = clock();
	cout<<"image filtering (smoothing) time is :"<<(double)(clock_3-clock_2)<<endl;
	

	findContours(thresh,contours,hierachy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
	//cout<<"contour"<<endl; 
	int cont_size=contours.size();
	for(int i=0;i<cont_size;i++)
	{
		double a=contourArea(contours[i],false);
		if(a>largest_area)
		{
			largest_area=a;
			largest_contour_index=i;
		}
	}

	vector<int> hull_index;
	Rect brect;

	if(largest_area>0 && contours[largest_contour_index].size()>5)
	{

		approxPolyDP(contours[largest_contour_index],contours[largest_contour_index],8,true);
		//cout<<"approx_poly"<<endl;
		convexHull(Mat(contours[largest_contour_index]),hull[0],false,true);
		//	cout<<"convex_hull"<<endl;
		brect=boundingRect(contours[largest_contour_index]);
		//cout<<"bounding_rect"<<endl;
		convexHull(Mat(contours[largest_contour_index]),hull_index,true);
		//cout<<"convex_hull2"<<endl;
		convexityDefects(contours[largest_contour_index], hull_index, defects);
		//cout<<"convexity defect"<<endl;
		// Mom ents mom=moments(contours[largest_contour_index]);
		// draw mass center
		//	circle(img,Point(mom.m10/mom.m00,mom.m01/mom.m00),2,cv::Scalar(0),2);


		Scalar colorw = Scalar(0,255,0);
		Scalar color1 = Scalar(0,0,255);
		//drawContours(img,contours,largest_contour_index,color,2, 8, hierachy);
		//drawContours(timg,contours,largest_contour_index,color,1, 8, hierachy);
		//drawContours(timg, hull, 0, color1, 1, 8, vector<Vec4i>(), 0, Point() );
		//	drawContours(img, hull, 0, color1, 2, 8, vector<Vec4i>(), 0, Point() );
		int defc_size=defects.size();

		Point ptStart;
		Point ptEnd;
		Point ptStart2;
		Point ptEnd2;
		Point ptFar;
		int count=1;
		int startidx2;
		int endidx2;
		int tolerance =  brect.height/5;
		float angleTol=95;
		for(int in=0;in<defc_size;in++)
		{
			//Vec4i& v=(*d); d++;
			int startidx=defects[in].val[0];ptStart=contours[largest_contour_index].at(startidx);
			int endidx=defects[in].val[1]; ptEnd=contours[largest_contour_index].at(endidx);
			int faridx=defects[in].val[2];  ptFar=contours[largest_contour_index].at(faridx);
			if(in+1<defc_size)
				startidx2=defects[in+1].val[0];ptStart=contours[largest_contour_index].at(startidx);
			endidx2=defects[in+1].val[1]; ptEnd=contours[largest_contour_index].at(endidx);

			if(distanceP2P(ptStart, ptFar) > tolerance && distanceP2P(ptEnd, ptFar) > tolerance && getAngle(ptStart, ptFar, ptEnd  ) < angleTol ){
				{
					if(in+1<defc_size)
					{
						if(distanceP2P(ptStart,ptEnd2) < tolerance )
							contours[largest_contour_index][startidx]=ptEnd2;
						else{
							if(distanceP2P(ptEnd,ptStart2) < tolerance )
								contours[largest_contour_index][startidx2]=ptEnd;

						}
					}
					defect_circle.push_back(ptFar);
				//	cout<<"ptfar"<<ptFar.x<<"&&"<<ptFar.y<<endl;

					if(count==1) 
					{
						FingerTips.push_back(ptStart);
						cv::circle(img1, ptStart,   2, Scalar(0,255,0 ), 2 );
						putText(img1,intToString(count),ptStart-Point(0,30),FONT_HERSHEY_PLAIN, 1.2f,Scalar(255,0,0),2);
					}
					FingerTips.push_back(ptEnd);
					count++;
					putText(img1,intToString(count),ptEnd-Point(0,30),FONT_HERSHEY_PLAIN, 1.2f,Scalar(255,0,0),2);
					cv::circle(img1, ptEnd,   2, Scalar(0,255,0 ), 2 );
					//cv::circle( img, ptFar,   2, Scalar(255,255,255 ), 2 );

				}
			}
		}
		//  circle(img, ptStart,2,Scalar(0xFF,0x60,0x02 ), 2, 8, 0 );

		//cv::circle( img, ptEnd,   4, Scalar( 0xFF,0x60,0x02 ), 2 );
			clock_t clock_4 = clock();
	cout<<"fingerTip detection  time is :"<<(double)(clock_4-clock_3)<<endl;

	//	cout<<"hii"<<endl;
		bool two_fn=false;
		bool five_fn=false;

		if(defect_circle.size()==1)
		{
			 two_fn=true;
			Point fn=FingerTips.back();
			FingerTips.pop_back();
			Point ln=FingerTips.back();
			FingerTips.pop_back();
			Point defect_point=defect_circle.back();
			float curr=getAngle(fn,defect_point,ln);
			curr=curr/10;
			curr=10-curr;
			renderBackgroundGL(img1);
				objectPoints1.push_back(Point3d(9,6,0));
			imagePoints1.push_back(defect_point);

			objectPoints1.push_back(Point3d(9,6,0));
			imagePoints1.push_back(defect_point);

		

			objectPoints1.push_back(Point3d(19,6,0));
			imagePoints1.push_back(fn);

			 objectPoints1.push_back(Point3d(9,18,0));
			   imagePoints1.push_back(ln);

			
			  // cout<<width<<"  &"<<height<<endl;
		//	cout<<"solvepnp"<<endl; 
			solvePnP(Mat(objectPoints1),Mat(imagePoints1),intrinsic_Matrix,distortion_coeffs,rvec,tvec);



			cv::Mat rotation;
			cv::Rodrigues(rvec, rotation);
			double offsetA[3][1] = {9,6,6};
			Mat offset(3, 1, CV_64FC1, offsetA);
			tvec = tvec + rotation*offset;


			generateProjectionModelview(intrinsic_Matrix, rotation, tvec, Projection, modelview);
			glMatrixMode(GL_PROJECTION);	
			GLfloat* projection = convertMatrixType(Projection);
			glLoadMatrixf(projection);
			delete[] projection;

			glMatrixMode(GL_MODELVIEW);
			GLfloat* modelView = convertMatrixType(modelview);
			glLoadMatrixf(modelView);
			delete[] modelView;
			
			//glTranslatef(0.0f,0.0f,-5.0f);
			glPushMatrix();
			glColor3f(1.0,0.0,0.0);
			
			glutWireTeapot(10.0/curr);
			glPopMatrix();
			glColor3f(1.0,1.0,1.0);

		}
		//Rotation Module
		if(defect_circle.size()==4)
		{

			five_fn=true;
			minEnclosingCircle(defect_circle,center,radius);
			//circle(img, center, (int)radius,Scalar(255,255,255), 2, 8, 0 );
			circle(img1, center,2,Scalar(0), 2, 8, 0 );

			vector<pair<float,Point>> pos;
			for(int in=0;in<FingerTips.size();in++)
			{
				Point p=FingerTips.back();
				FingerTips.pop_back();

				//if(in==0)
				//{
				pos.push_back(make_pair(distanceP2P(center,p),p));
				//position.push_back(pos);
			}
				//	}
				//else
				//	{
				//		cout<<"size is"<<position.size()<<endl;
				//	position[n].push_back(make_pair(distanceP2P(center,p),p));
				//}
			
			sort(pos.begin(),pos.end(),pairCompare);
			//	vector<pair<float,Point>> now=position[i].back();
			first=pos.back().second;
			pos.pop_back();
			//cout<<"new value :"<<new1.x<<" && "<<new1.y<<endl;
			second=pos.back().second;
			pos.pop_back();
			third=pos.back().second;
			pos.pop_back();
			
			if(third.y<second.y&&second.y<first.y)
			{
			//	cout<<"vertical pose"<<endl;
				FIX_X.x=center.x+40;
				FIX_X.y=center.y;

				FIX_Y.x=center.x;
				FIX_Y.y=center.y-40;
			}
			skew_x=getAngle(first,center,FIX_X);
			skew_y=getAngle(third,center,FIX_Y);
			cout<<skew_x<<"&"<<skew_y<<endl;
			if(first.x<img1.cols)
			line(img1,center,first,Scalar(200,200,200),2,8,0);
			line(img1,center,FIX_X,Scalar(200,200,200),2,8,0);
			if(second.x<img1.cols)
				line(img1,center,second,Scalar(0,255,0),2,8,0);
				if(third.x<img1.cols)
					line(img1,center,third,Scalar(0,0,255),2,8,0);
				line(img1,center,FIX_Y,Scalar(0,0,255),2,8,0);
		
			//	line(img1,center,first,Scalar(255,255,255),2,8,0);
			//	line(img1,center,second,Scalar(0,255,255),2,8,0);
			//	line(img1,center,third,Scalar(0,0,255),2,8,0);
		
		renderBackgroundGL(img1);

			/*	cvtColor(test, gray1, CV_RGB2GRAY);
			bool sCorner1=findChessboardCorners(gray1, Size(7, 7), corners1);
			imshow("live",test);
			if(sCorner1)
			{
			cornerSubPix(gray1, corners1, Size(11,11), Size(-1,-1), TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
			if(corners1.size() == 7*7)
			{control pan

			for(int j=0; j< corners1.size(); ++j)
			{
			Point2f tImgPT;
			Point3f tObjPT;

			tImgPT.x = corners1[j].x;
			tImgPT.y = corners1[j].y;

			tObjPT.x = j%7*3;
			tObjPT.y = j/7*3;
			tObjPT.z = 0;
			imagePoints1.push_back(tImgPT);
			objectPoints1.push_back(tObjPT);     
			}

			vector<Point2f> projectedPoints;
			vector<Point3f> axis;

			axis.push_back(Point3f(6,0,0));
			axis.push_back(Point3f(0,6,0));
			axis.push_back(Point3f(0,0,6));  */
			objectPoints1.push_back(Point3d(9,6,0));
			imagePoints1.push_back(center);

			objectPoints1.push_back(Point3d(9,18,0));
			imagePoints1.push_back(first);

			objectPoints1.push_back(Point3d(19,6,0));
			imagePoints1.push_back(third);

			 objectPoints1.push_back(Point3d(15,15,0));
			   imagePoints1.push_back(second);

			
			  // cout<<width<<"  &"<<height<<endl;
		//	cout<<"solvepnp"<<endl; 
			solvePnP(Mat(objectPoints1),Mat(imagePoints1),intrinsic_Matrix,distortion_coeffs,rvec,tvec);



			cv::Mat rotation;
			cv::Rodrigues(rvec, rotation);
			double offsetA[3][1] = {9,6,0};
			Mat offset(3, 1, CV_64FC1, offsetA);
			tvec = tvec + rotation*offset;


			generateProjectionModelview(intrinsic_Matrix, rotation, tvec, Projection, modelview);

			/*  double offsetA[3][1] = {{(7-1.0)/2.0}, {(7-1.0)/2.0}, {0}};
			Mat offset(3, 1, CV_64FC1, offsetA);
			tvec = tvec + rotation*offset;

			for(unsigned int row=0; row<3; ++row)
			{
			for(unsigned int col=0; col<3; ++col)
			{
			modelview.at<float>(row, col) = rotation.at<float>(row, col);
			cout<<modelview.at<float>(row,col)<<endl;
			}
			modelview.at<float>(row, 3) = tvec.at<float>(row, 0);
			}
			modelview.at<float>(3, 3) = 1.0f;
			cout<<endl;



			static float changeCoordArray[4][4] = {{-1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};
			static Mat changeCoord(4, 4, CV_64FC1, changeCoordArray);

			modelview = changeCoord*modelview;

			cv::Mat glmodelview = cv::Mat::zeros(4, 4, CV_64F);
			transpose(modelview , glmodelview);  
			gluLookAt(0.0,2.0,-50.0,0.0,0.5,0.0,0.0,1.0,0.0);
			/*	glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			float fx=intrinsic_Matrix.at<float>(0,0);
			float fy=intrinsic_Matrix.at<float>(1,1);
			float cf=(2*atanf(0.5*height/fy)*180/3.14);
			float aspect=(width*fy)/(height*fx);   

			//gluPerspective(cf,1.0, zNear, zFar);   



			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glLoadMatrixf(&glmodelview.at<float>(0,0));  */
			glMatrixMode(GL_PROJECTION);	
			GLfloat* projection = convertMatrixType(Projection);
			glLoadMatrixf(projection);
			delete[] projection;

			glMatrixMode(GL_MODELVIEW);
			GLfloat* modelView = convertMatrixType(modelview);
			glLoadMatrixf(modelView);
			delete[] modelView;
			
			//glTranslat ef(0.0f,0.0f,-5.0f);
			glPushMatrix();
			glColor3f(1.0,0.0,0.0);
			glRotatef(skew_x,1.0,0.0,0.0);
			glRotatef(skew_y,0.0,1.0,0.0);
			glutWireTeapot(10.0);
			glPopMatrix();
			glColor3f(1.0,1.0,1.0);
			clock_t clock_5 = clock();
	cout<<"interaction time is :"<<(double)(clock_5-clock_4)<<endl;
		}
		imshow("live",img1);
		cout<<"----------------------------------------------"<<endl;

		glFlush();
		glutSwapBuffers();
	}
	waitKey(27);
	glutPostRedisplay();
}
void reshape(int x,int y)
{
	width=x; height=y;
	glViewport(0,0,width,height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	/*Projection.at<float>(0,0) = 2*intrinsic_Matrix.at<float>(0,0)/width;
	Projection.at<float>(1,0) = 0;
	Projection.at<float>(2,0) = 0;
	Projection.at<float>(3,0) = 0;

	Projection.at<float>(0,1) = 0;
	Projection.at<float>(1,1) = 2*intrinsic_Matrix.at<float>(1,1)/height;
	Projection.at<float>(2,1) = 0;
	Projection.at<float>(3,1) = 0;

	Projection.at<float>(0,2) = 1-2*intrinsic_Matrix.at<float>(0,2)/width;
	Projection.at<float>(1,2) = -1+(2*intrinsic_Matrix.at<float>(1,2)+2)/height;
	Projection.at<float>(2,2) = (zNear+zFar)/(zNear - zFar);
	Projection.at<float>(3,2) = -1;

	Projection.at<float>(0,3) = 0;
	Projection.at<float>(1,3) = 0;
	Projection.at<float>(2,3) = 2*zNear*zFar/(zNear - zFar);
	Projection.at<float>(3,3) = 0; 

	cv::Mat projection = cv::Mat::zeros(4, 4, CV_64F);
	transpose(Projection ,projection);  
	glLoadMatrixf(&projection.at<float>(0,0));   */


	// gluPerspective(60, (GLfloat)width / (GLfloat)height, 1.0, 100.0); 

	/* float fx=intrinsic_Matrix.at<float>(0,0);
	float fy=intrinsic_Matrix.at<float>(1,1);
	float cf=(2*atanf(0.5*height/fy)*180/3.14);
	cout<<fx<<"   "<<fy<<endl;
	float aspect=(width*fy)/(height*fx);  */
	//gluPerspective(cf,CALIB_FIX_ASPECT_RATIO, zNear, zFar); 


	//glMatrixMode(GL_MODELVIEW);
	//	gluPerspective(60,width/height, zNear, zFar);  
	//glOrtho(-100,100,-100.0,100,zNear, zFar);

}
void init()
{
	glClearColor(0.0f,0.0f,0.0f,0.0f);
}
void main()
{
	if(!cap.isOpened())
	{
		exit(-1);
	}
	cap >> test;
	if(!test.data)
	{
		exit(-1);
	}
//	resize(test,test,Size(180,180),0,0);
	width=test.cols;
	height=test.rows;
	cout<<width<<endl;
	calibrate(intrinsic_Matrix,distortion_coeffs);
	glutInitDisplayMode(GLUT_RGB |GLUT_DOUBLE |GLUT_DEPTH );
	glutInitWindowSize(width,height);
	glutCreateWindow("code4change");
	init();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMainLoop();
}

