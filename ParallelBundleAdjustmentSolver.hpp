#include<iostream>
#include<vector>
#include<Eigen/Sparse>
#include<Eigen/Eigen>
#include<boost/mpi.hpp>
#include"DataType.hpp"

#define EIGEN_NO_DEBUG
#define BUNDLER_MODE
#define DENSE_MATRIX_STORAGE
#define COLUMN_BLOCK_SPARSE_STORAGE
#define COMPRESSED_STORAGE

extern boost::mpi::environment env;
extern boost::mpi::communicator world;


class ParallelBundleAdjustmentSolver{
	private:
		//concurrent computing, global size and current rank
		int procSize;
		int procRank;

		int maxIteration;
		int mBlockSize;

#ifdef BUNDLER_MODE
		int structureDimen;
		int cameraDimen;  //including intrinsic and extrinsic parameters, k2 is omitted here
#else
		int structureDimen;
		int cameraIntrinsicDimen;
		int cameraExtrinsicDimen;
#endif

		unsigned int numOfObjectPoint;
		unsigned int numOfProjection;
		unsigned int numOfCamera;
		unsigned int localObjectPointSize;
		unsigned int localObservationSize;
		unsigned int localCameraSize;
		std::vector<Eigen::MatrixXd> structureMatrix;
		std::vector<Eigen::MatrixXd> observationMatrix;
		Eigen::VectorXi osvColIdx;
		Eigen::VectorXi osvRowIdx;
		std::vector<Eigen::MatrixXd> cameraMatrix;
		int cameraMatrixSize;
		Eigen::VectorXd objectPointVector;
		Eigen::VectorXd cameraVector;
		Eigen::VectorXd objectPointVectorX;
		Eigen::VectorXd cameraVectorX;


		bool bFocalNormalize;
		bool bDepthNormalize;
		bool bUseRadialDistortion;
		double _data_normalize_median;
		double _focal_scaling;
		double _depth_scaling;
		double MSE;
		double criticalMSE;

		//return false when normalMatrix is not symmetrical and positive definite
		bool fillSparseMatrix(std::vector<ObjectPoint>& vObjectPointList, std::vector<Projection>& vProjectionList, std::vector<Camera>& vCameraList);
		void parallelSchurComplement(Eigen::MatrixXd& RCM, Eigen::VectorXd& RCV);
		void parallelCholeskyDecomposition(Eigen::MatrixXd& RCM, Eigen::VectorXd& RCV);
		void parallelForwardBackwardSubstitution(Eigen::MatrixXd& RCM, Eigen::VectorXd& RCV);
		void updateParameters(std::vector<ObjectPoint>& vObjectPointList, std::vector<Projection>& vProjectionList, std::vector<Camera>& vCameraList);
		void setMatrixSize(unsigned int objectPointMatrixSize, unsigned int observationMatrixSize, unsigned int cameraMatrixSize);

	public:
		ParallelBundleAdjustmentSolver(int procRank, int procSize);
		~ParallelBundleAdjustmentSolver();

		void setDimensionParameters(unsigned int numOfObjectPoint, unsigned int numOfProjection, unsigned int numOfCamera, 
					unsigned int localObjectPointSize, unsigned int localObservationSize, unsigned int localCameraSize);
		void normalizeFocal(std::vector<Camera> vCameraList);
		void normalizeDepth();
		void dumpHessianMatrixEntry(std::vector<ObjectPoint>& vObjectPointList, std::vector<Projection>& vProjectionList, std::vector<Camera>& vCameraList);
		void parallelBundleAdjustment(std::vector<ObjectPoint>& vObjectPointList, std::vector<Projection>& vProjectionList, std::vector<Camera>& vCameraList);




};
