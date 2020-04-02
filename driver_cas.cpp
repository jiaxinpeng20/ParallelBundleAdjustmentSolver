#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include"ParallelBundleAdjustmentSolver.hpp"

boost::mpi::environment env;
boost::mpi::communicator world;

bool loadBundlerDataFromFile(std::ifstream& in, std::vector<ObjectPoint>& vObjectPointList, std::vector<Projection>& vProjectionList, std::vector<Camera>& vCameraList, unsigned int& numOfObjectPoint, unsigned int& numOfProjection, unsigned int& numOfCamera)
{
	
	if(!(in>>numOfCamera>>numOfObjectPoint>>numOfProjection))
		return false; 

	unsigned int localObjectPointNum = 0;
	unsigned int localProjectionNum = 0;
	unsigned int localCameraNum = 0;

	//vCameraList.resize(localCameraNum);
	//vProjectionList.resize(localProjectionNum);
	//vObjectPointList.resize(localObjectPointNum );


	if(world.rank() == 0)
	{
		std::cout<<"Reading objectpoint and observation data in a crossover way."<<std::endl;
		std::cout<<"Complete camera data have to be read."<<std::endl;
	}

	for(unsigned int i = 0; i < numOfProjection; i++)
	{
		int camidx, optidx;
		double x, y;

		if(!(in>>camidx>>optidx>>x>>y))
			return false;

		if(optidx%world.size() == world.rank())
		{
			Projection node;
			node.setProjectionParameter(camidx, optidx, x, y);
			vProjectionList.push_back(node);

			localProjectionNum++;
		}
	}


	for(unsigned int i = 0; i < numOfCamera; i++)
	{
		double p[9];
		for(int j = 0; j < 9; j++) in>>p[j];

		Camera node;		
		node.setInvertedRT(p, p+3);
		node.setFocalLength(p[6]);

		node.setProjectionDistortion(p[7], p[8]);
		vCameraList.push_back(node);

		localCameraNum++;
	}

	for(unsigned int i = 0; i < numOfObjectPoint; i++)
	{
		double p0, p1, p2;
		in>>p0>>p1>>p2;
		
		if(i%world.size() == world.rank())
		{
			ObjectPoint node;
			node.setPoint(p0, p1, p2);
			vObjectPointList.push_back(node);

			localObjectPointNum++;
		}
	}


	std::cout<<"[CPU#"<<world.rank()<<"]:"<<"Number of objectpoints: "<<localObjectPointNum<<std::endl;
	std::cout<<"[CPU#"<<world.rank()<<"]:"<<"Number of projections: "<<localProjectionNum<<std::endl;
	std::cout<<"[CPU#"<<world.rank()<<"]:"<<"Number of cameras: "<<localCameraNum<<std::endl;

	return true;
	
}

bool loadBundlerDataFromFileV3(std::ifstream& in, std::vector<ObjectPoint>& vObjectPointList, std::vector<Projection>& vProjectionList, std::vector<Camera>& vCameraList, unsigned int& numOfObjectPoint, unsigned int& numOfProjection, unsigned int& numOfCamera)
{
	std::cout<<"Is this procedure correct?"	<<std::endl;
	std::string drop;
	in>>drop>>drop>>drop>>drop;	
	if(!(in>>numOfCamera>>numOfObjectPoint))
		return false; 

	unsigned int localObjectPointNum = 0;
	unsigned int localProjectionNum = 0;
	unsigned int localCameraNum = 0;

	//vCameraList.resize(localCameraNum);
	//vProjectionList.resize(localProjectionNum);
	//vObjectPointList.resize(localObjectPointNum );


	if(world.rank() == 0)
	{
		std::cout<<"Reading objectpoint and observation data in a crossover way."<<std::endl;
		std::cout<<"Complete camera data have to be read."<<std::endl;
	}

	/*for(unsigned int i = 0; i < numOfProjection; i++)
	{
		int camidx, optidx;
		double x, y;

		if(!(in>>camidx>>optidx>>x>>y))
			return false;

		if(optidx%world.size() == world.rank())
		{
			Projection node;
			node.setProjectionParameter(camidx, optidx, x, y);
			vProjectionList.push_back(node);

			localProjectionNum++;
		}
	}*/


	for(unsigned int i = 0; i < numOfCamera; i++)
	{
		double p[9];
		Camera node;		
		for(int j = 0; j < 3; j++) in>>p[j];

		node.setFocalLength(p[0]);
		node.setProjectionDistortion(p[1], p[2]);
		
		for(int j = 0; j < 9; j++) in>>p[j];
		node.setMatrixRotation(p);

		for(int j = 0; j < 3; j++) in>>p[j];
		node.setTranslation(p);

		vCameraList.push_back(node);

		localCameraNum++;

		/*std::cout<<node.getFocalLength()<<std::endl;
		std::cout<<node.getDistortionk1()<<"  "<<node.getDistortionk2()<<std::endl;
		std::cout<<node.getRotation()<<std::endl;
		std::cout<<node.getTranslation()<<std::endl;*/  

	}

	//processing the additional line
	double p[3];
	for(int j = 0; j < 3; j++) in>>p[j];

	//read object points from file, and extract projection data
	for(unsigned int i = 0; i < numOfObjectPoint; i++)
	{
		int projectionNum;
		int colorR, colorG, colorB;
		in>>colorR>>colorG>>colorB;
		in>>projectionNum;

		int camIndex, projectIndex;
	       	double	x, y;
		for(unsigned int j = 0; j < projectionNum; j++)
		{
			in>>camIndex>>projectIndex>>x>>y; 

			if(i%world.size() == world.rank())
			{
				Projection node;
				node.setProjectionParameter(camIndex, i, x, y);
				vProjectionList.push_back(node);

				localProjectionNum++;

			}
		}
	
		double p0, p1, p2;	
		in>>p0>>p1>>p2;
		
		if(i%world.size() == world.rank())
		{
			ObjectPoint node;
			node.setPoint(p0, p1, p2);
			vObjectPointList.push_back(node);

			localObjectPointNum++;
		}

	}


	std::cout<<"[CPU#"<<world.rank()<<"]:"<<"Number of objectpoints: "<<localObjectPointNum<<std::endl;
	std::cout<<"[CPU#"<<world.rank()<<"]:"<<"Number of projections: "<<localProjectionNum<<std::endl;
	std::cout<<"[CPU#"<<world.rank()<<"]:"<<"Number of cameras: "<<localCameraNum<<std::endl;

	return true;
	
}

void dumpSparsePointCloud(std::vector<ObjectPoint> vObjectPointList, std::vector<Projection> vProjectionList, std::vector<Camera> vCameraList)
{
	std::ofstream out("./sparse_point_cloud_cas.ply");
	out<<"ply"<<std::endl;
	out<<"format ascii 1.0"<<std::endl;
	out<<"element vertex "<<vObjectPointList.size()<<std::endl;
	out<<"property double x"<<std::endl;
	out<<"property double y"<<std::endl;
	out<<"property double z"<<std::endl;
	out<<"end_header"<<std::endl;

	for(int i = 0; i < vObjectPointList.size(); i++)
	{
		out<<vObjectPointList[i].getPointX()<<std::endl;
		out<<vObjectPointList[i].getPointY()<<std::endl;
		out<<vObjectPointList[i].getPointZ()<<std::endl;
	}

	out.close();
}
	
bool storeBundlerDataToFile(std::ofstream& out, std::vector<ObjectPoint>& vObjectPointList, std::vector<Projection>& vProjectionList, std::vector<Camera>& vCameraList)
{

}

int main(int argc, char* argv[])
{
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	std::vector<Camera> vCameraList;
	std::vector<ObjectPoint> vObjectPointList;
	std::vector<Projection> vProjectionList;

	unsigned int numOfObjectPoint;
	unsigned int numOfCamera;
	unsigned int numOfProjection;

	std::ifstream in(argv[1]);
	if(!in.is_open())
	{
		std::cout<<"Open file failed, check your input!"<<std::endl;
		return -1;
	}

	if(!loadBundlerDataFromFileV3(in, vObjectPointList, vProjectionList, vCameraList, numOfObjectPoint, numOfProjection, numOfCamera))
	{
		std::cout<<"Load bundler data from file failed!!!"<<std::endl;
	}

	ParallelBundleAdjustmentSolver solver(world.rank(), world.size());
	solver.setDimensionParameters(numOfObjectPoint, numOfProjection, numOfCamera, vObjectPointList.size(), vProjectionList.size(), vCameraList.size());
	solver.parallelBundleAdjustment(vObjectPointList, vProjectionList, vCameraList);
	//dumpSparsePointCloud(vObjectPointList, vProjectionList, vCameraList);
	return 0;
}
