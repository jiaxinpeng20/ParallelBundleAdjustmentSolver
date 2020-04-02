#include<iostream>
#include<stdlib.h>
#include<Eigen/Eigen>
#include<boost/mpi.hpp>
#include<vector>
#include<math.h>

//boost::mpi::environment env;
//boost::mpi::communicator world;

using namespace Eigen;


void ParallelCholeskyDecomposition(boost::mpi::communicator world, double* RCM, int capacity, int cameraMatrixSize, int mBlockSize)
{
	for(int i = 0; i < cameraMatrixSize/6; i++) //for rows
	{
		MatrixXd mMatrixDiag(6, 6);

		if(i%world.size() == world.rank())
		{
			int blockOffset = (world.rank()+1 + i+1) * ((i-world.rank())/world.size() + 1) / 2;
			blockOffset = (blockOffset-1) * 36;

			for(int j = 0; j < 6; j++)
			for(int k = 0; k < 6; k++)
			{
				mMatrixDiag.coeffRef(j,k) = RCM[blockOffset + j*6 + k];
			}

			//if(world.rank()==0) std::cout<<mMatrixDiag<<std::endl;
			mMatrixDiag = mMatrixDiag.llt().matrixU();
			//if(world.rank()==0)std::cout<<mMatrixDiag<<std::endl;

			//write this block back
			for(int j = 0; j < 6; j++)
			for(int k = 0; k < 6; k++)
			{
				RCM[blockOffset + j*6 + k] = mMatrixDiag.coeffRef(j,k);
			}

		}

		boost::mpi::broadcast(world, mMatrixDiag.data(), 36, i%world.size());

		//compute submatrices for the i-th row
		int start = (i%world.size() < world.rank() ? i/world.size() * world.size()+world.rank() : i/world.size()*world.size()+world.size()+world.rank());
		for(int j = start; j < cameraMatrixSize/mBlockSize; j+=world.size())
		{
			MatrixXd temp(6, 6);

			int blockOffset = 0;
			for(int k = world.rank(); k < j; k+=world.size())
				blockOffset += k+1;
			
			blockOffset = (blockOffset + i)*36;
			for(int k = 0; k < 6; k++)
			for(int l = 0; l < 6; l++)
			{
				temp.coeffRef(k,l) = RCM[blockOffset + k*6 +l];
			}

			temp = MatrixXd(MatrixXd(mMatrixDiag.transpose()).inverse()) * temp;
			for(int k = 0; k < 6; k++)
			for(int l = 0; l < 6; l++)
			{
				RCM[blockOffset + k*6 + l];
			}
		}

		//update the rest rows
		MatrixXd pasadena((i+1)*6, 6);
		if((i+1)%world.size() == world.rank())
		{
			int blockOffset = 0;
			for(int k = world.rank(); k < i+1; k+=world.size())
				blockOffset += k+1;

			blockOffset = blockOffset*36;
			//fill matrix pasadena
			for(int j = 0; j < (i+1)*6; j++)
			for(int k = 0; k < 6; k++)
			{
				pasadena.coeffRef(j,k) = RCM[blockOffset + j*6 + k];
			}


		}
		boost::mpi::broadcast(world, pasadena.data(), (i+1)*36, (i+1)%world.size());

		for(int j = start; j < cameraMatrixSize/6; j+=world.size())
		{
			int blockOffset = 0;
			for(int k = world.rank(); k < j; k+=world.size())
				blockOffset += k+1;


			blockOffset = blockOffset*36;
			MatrixXd base((i+1)*6, 6);
			for(int k = 0; k < (i+1)*6; k++)
			for(int l = 0; l < 6; l++)
			{
				base.coeffRef(k,l) = RCM[blockOffset + k*6 + l];
			}

			MatrixXd temp(6, 6);
			for(int k = 0; k < 6; k++)
			for(int l = 0; l < 6; l++)
			{
				temp.coeffRef(k,l) = RCM[blockOffset + (i+1)*36 + k*6 +l];
			}
			temp -= MatrixXd(pasadena.transpose()) * base;

			for(int k = 0; k < 6; k++)
			for(int l = 0; l < 6; l++)
			{
				RCM[blockOffset + (i+1)*36 + k*6 + l] = temp.coeffRef(k,l);
			}
		}
		

		world.barrier();
		i++;
		world.barrier();
	}
}




int main(int argc, char* argv[])
{
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	//create normal matrix by random
	int objectpoint = 20000;
	int image = 180;
	int camera = 5;

	int objectpointLocal = objectpoint/world.size() + (objectpoint%world.size()>world.rank()?1:0);
	int imageLocal = image/world.size() + (image%world.size()>world.rank()?1:0);
	
	int cameraMatrixSize = image*6 + camera*6;

	//fill Objectpoint Matrix
	std::vector<Eigen::Triplet<double>> triple;
	Eigen::SparseMatrix<double> objectpointMatrix(objectpointLocal*3, objectpointLocal*3);
	for(int i = 0; i < objectpointLocal; i++)
	{
		for(int j = 0; j < 3; j++)
		for(int k = j ; k < 3; k++)
		{
			int row = i*3 + j;
			int col = i*3 + k;
			double val = ((double)rand()/RAND_MAX)*(100 - 0.00001) + 0.00001;
			triple.push_back(Eigen::Triplet<double>(row, col, val));
		}
		
	}
	objectpointMatrix.setFromTriplets(triple.begin(), triple.end());
	
	//fill Observation Matrix, observation by images and observations by cameras
	triple.clear();
	Eigen::SparseMatrix<double> observationMatrix(objectpointLocal*3, cameraMatrixSize);
	double sparseRate = 0.05;
	for(int i = 0; i < objectpointLocal; i++)
	{
		srand(i);
		int observationCount = round((camera+image) * sparseRate);

		
		std::vector<int> linkage;
		for(int j = 0; j < observationCount; j++)
		{
			srand(i+j);
			linkage.push_back(random()%(image));
		}
		for(int j = image; j < camera+image; j++)
		{
			linkage.push_back(j);
		}

		for(std::vector<int>::iterator iter = linkage.begin(); iter != linkage.end(); iter++)
		{
			for(int j = 0; j < 3; j++)
			for(int k = 0; k < 6; k++)
			{
				int row = i*3 + j;
				int col = (*iter)*6 + k;
				srand(row*cameraMatrixSize + col);
				double val = ((double)rand()/RAND_MAX)*(100 - 0.00001) + 0.00001;
				triple.push_back(Eigen::Triplet<double>(row, col, val));
			}
		}

	}
	observationMatrix.setFromTriplets(triple.begin(), triple.end());

	Eigen::VectorXd objectpointVector(objectpointLocal*3);
	for(int i = 0; i < objectpointLocal*3; i++)
	{
		srand(i);
		objectpointVector(i) = ((double)rand()/RAND_MAX)*(100 - 0.00001) + 0.00001;
	}
	//std::cout<<observationMatrix.leftCols(30)<<std::endl;


	//fill Camera Matrix, with extrinsic parameters and intrinsic parameters
	triple.clear();
	Eigen::SparseMatrix<double> cameraMatrix(cameraMatrixSize, cameraMatrixSize);
	for(int i = 0; i < camera+image; i++)//in diagonal
	{
		for(int j = 0; j < 6; j++)
		for(int k = j; k < 6; k++)
		{
			int row = i*6 + j;
			int col = i*6 + k;
			srand(row*cameraMatrixSize + col);
			double val = ((double)rand()/RAND_MAX)*(100 - 0.00001) + 0.00001;
			triple.push_back(Eigen::Triplet<double>(row, col, val));
		}
	}
	double captureRate = 0.5;
	for(int i = 0; i <image; i++ )
	{
		srand(i);
		int captureCount = round(camera * captureRate);

		std::vector<int> linkage;
		for(int j = 0; j < captureCount; j++)
		{
			srand(i+j);
			linkage.push_back(rand()%camera + image);
		}
		for(std::vector<int>::iterator iter = linkage.begin(); iter != linkage.end(); iter++)
		{
			for(int j = 0; j < 6; j++)
			for(int k = 0; k < 6; k++)
			{
				int row = 6*i + j;
				int col = (*iter)*6 + k;
				srand(row*cameraMatrixSize + col);
				double val = ((double)rand()/RAND_MAX)*(100 - 0.00001) + 0.00001;
				triple.push_back(Eigen::Triplet<double>(row, col, val));
			}
		}
	}
	cameraMatrix.setFromTriplets(triple.begin(), triple.end());
	Eigen::VectorXd cameraVector(objectpointLocal*3);
	for(int i = 0; i < cameraMatrixSize; i++)
	{
		srand(i);
		cameraVector(i) = ((double)rand()/RAND_MAX)*(100 - 0.00001) + 0.00001;
	}
	//std::cout<<cameraMatrix.leftCols(30)<<std::endl;
	
	//Schur Complement
	boost::mpi::timer timing;
	Eigen::SparseMatrix<double> objectpointMatrixInv = objectpointMatrix;
	for(int i = 0; i < objectpointLocal*3; i+=3)
	{
		Eigen::MatrixXd temp = objectpointMatrix.block(i, i, 3, 3);
		temp = Eigen::MatrixXd(temp.selfadjointView<Eigen::Upper>()).inverse();

		for(int j = i; j < i+3; j++)
		for(int k = j; k < i+3; k++)
		{
			objectpointMatrixInv.coeffRef(j, k) = temp(j-i, k-i);
		}
	}
	Eigen::SparseMatrix<double> S = observationMatrix.transpose() * objectpointMatrixInv.selfadjointView<Eigen::Upper>();
	Eigen::SparseMatrix<double> MT = S * observationMatrix;
	Eigen::VectorXd VT = S * objectpointVector;
	double t0 = timing.elapsed();

	//Gathering and Sampling 
	timing.restart();
	int mLine = 1000;
	SparseMatrix<double> sumMatrix(cameraMatrixSize, cameraMatrixSize);
	for(int i = 0; i <= cameraMatrixSize/mLine; i++)
	{
		if(i < cameraMatrixSize/mLine)
		{
			Eigen::MatrixXd MTReduce(cameraMatrixSize, mLine);
			boost::mpi::all_reduce(world, MT.middleCols(i*mLine, mLine).toDense().data(), cameraMatrixSize*mLine, MTReduce.data(), std::plus<double>());
			sumMatrix.middleCols(i*mLine, mLine) = MTReduce.sparseView();
		}
		else
		{
			Eigen::MatrixXd MTReduce(cameraMatrixSize, cameraMatrixSize%mLine);
			boost::mpi::all_reduce(world, MT.middleCols(i*mLine, cameraMatrixSize%mLine).toDense().data(), cameraMatrixSize*(cameraMatrixSize%mLine), MTReduce.data(), std::plus<double>());
			sumMatrix.middleCols(i*mLine, cameraMatrixSize%mLine) = MTReduce.sparseView();
		}
	}

	sumMatrix = sumMatrix.triangularView<Eigen::Upper>();
	sumMatrix = sumMatrix - cameraMatrix; //inverse type, for cholesky decomposition

	VectorXd VTReduce(cameraMatrixSize);
	boost::mpi::all_reduce(world, VT.data(), cameraMatrixSize, VTReduce.data(), std::plus<double>());

	int mBlockSize = 6;
	int columnBlockCount = cameraMatrixSize/mBlockSize;
	int res = cameraMatrixSize%mBlockSize;

	int capacity = 0;
	for(int i = world.rank(); i < columnBlockCount; i+=world.size())
	{
		capacity += (i+1);
	}
	capacity = capacity * mBlockSize*mBlockSize;
	if((res!=0)&&(columnBlockCount+1)%mBlockSize-1 == world.rank())
	{
		capacity = capacity + res*cameraMatrixSize;
	}

	double* RCM = (double*)malloc(capacity*sizeof(double));//fill submatrix of every process in a compressed format
	int index = 0;
	for(int i = world.rank(); i <= cameraMatrixSize/mBlockSize; i+=world.size())
	{
		int margin = (i < cameraMatrixSize/mBlockSize? mBlockSize: cameraMatrixSize%mBlockSize);
		if(margin == 0) break;

		Eigen::MatrixXd temp = sumMatrix.block(0, i*mBlockSize, (i+1)*mBlockSize, margin);

		for(int j = 0; j < (i+1)*mBlockSize; j++)
		for(int k = 0; k < margin; k++)
		{
			RCM[index++] = temp.coeffRef(j, k);
		}
	}

	double t1 = timing.elapsed();
	//delete sumMatrix;




	//Parallel Cholesky Decomposition
	timing.restart();
	ParallelCholeskyDecomposition(world, RCM, capacity, cameraMatrixSize, mBlockSize);




	double t2 = timing.elapsed();
	world.barrier();
	if(world.rank()==0)
	{
		std::cout<<"Feature Points:    "<< objectpoint<<std::endl;
		std::cout<<"Images Related:    "<<image<<std::endl;
		std::cout<<"Camera Related:    "<<camera<<std::endl;
		std::cout<<"Hessian Matrix:    "<<objectpoint*3+image*6+camera*6<<std::endl;
		std::cout<<"Time Record:"<<std::endl;
		std::cout<<"Parallel Shur Complement     "<<"Gathering and Sampling    "<<"Parallel Cholesky Decomposition    "<<"Global Time Consumed"<<std::endl;
	}
	world.barrier();
		std::cout<<"       "<<t0<<"               "<<"       "<<t1<<"                        "<<t2<<"                     "<<t0+t1+t2<<std::endl;

	return 0;
}
