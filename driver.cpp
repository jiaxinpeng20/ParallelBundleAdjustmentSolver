#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include"ParallelBundleAdjustmentSolver.hpp"

boost::mpi::environment env;
boost::mpi::communicator world;

bool loadBundlerDataFromFile(std::ifstream& in, std::vector<ObjectPoint>& vObjectPointList, std::vector<Projection>& vProjectionList, std::vector<Camera>& vCameraList,
				unsigned int& numOfObjectPoint, unsigned int& numOfProjection, unsigned int& numOfCamera)
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

bool storeBundlerDataFromFile(std::ofstream& out, std::vector<ObjectPoint>& vObjectPointList, std::vector<Projection>& vProjectionList, std::vector<Camera>& vCameraList)
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

	if(!loadBundlerDataFromFile(in, vObjectPointList, vProjectionList, vCameraList, numOfObjectPoint, numOfProjection, numOfCamera))
	{
		std::cout<<"Load bundler data from file failed!!!"<<std::endl;
	}

	ParallelBundleAdjustmentSolver solver(world.rank(), world.size());
	solver.setDimensionParameters(numOfObjectPoint, numOfProjection, numOfCamera, vObjectPointList.size(), vProjectionList.size(), vCameraList.size());
	solver.parallelBundleAdjustment(vObjectPointList, vProjectionList, vCameraList);

	return 0;
}
