#include<math.h>

class Camera{
private:
	double T[3];
	double R[3][3];
	double focalLength;//focal length
	double k1;         // It seems that pba only use k1 parameters for radial distortion
	double k2;

public:
	Camera()
	{
	}

	~Camera()
	{
	}

	void setRodriguesRotation(const double r[3])
	{
        	double a = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        	double ct = a==0.0?0.5:(1.0-cos(a))/a/a;
        	double st = a==0.0?1:sin(a)/a;
        	R[0][0]=double(1.0 - (r[1]*r[1] + r[2]*r[2])*ct);
        	R[0][1]=double(r[0]*r[1]*ct - r[2]*st);
        	R[0][2]=double(r[2]*r[0]*ct + r[1]*st);
        	R[1][0]=double(r[0]*r[1]*ct + r[2]*st);
        	R[1][1]=double(1.0 - (r[2]*r[2] + r[0]*r[0])*ct);
        	R[1][2]=double(r[1]*r[2]*ct - r[0]*st);
        	R[2][0]=double(r[2]*r[0]*ct - r[1]*st);
        	R[2][1]=double(r[1]*r[2]*ct + r[0]*st);
        	R[2][2]=double(1.0 - (r[0]*r[0] + r[1]*r[1])*ct );
    }

	void getRodriguesRotation(double r[3]) const
    {
        double a = (R[0][0]+R[1][1]+R[2][2]-1.0)/2.0;
        const double epsilon = 0.01;
        if( fabs(R[0][1] - R[1][0]) < epsilon &&
            fabs(R[1][2] - R[2][1]) < epsilon && 
            fabs(R[0][2] - R[2][0]) < epsilon )
        {
            if( fabs(R[0][1] + R[1][0]) < 0.1 &&
                fabs(R[1][2] + R[2][1]) < 0.1 && 
                fabs(R[0][2] + R[2][0]) < 0.1 && a > 0.9)
            {
                r[0]    =    0;
                r[1]    =    0;
                r[2]    =    0;
            }
            else
            {
                const double ha = sqrt(0.5) * 3.14159265358979323846; 
                double xx = (R[0][0]+1.0)/2.0;
                double yy = (R[1][1]+1.0)/2.0;
                double zz = (R[2][2]+1.0)/2.0;
                double xy = (R[0][1]+R[1][0])/4.0;
                double xz = (R[0][2]+R[2][0])/4.0;
                double yz = (R[1][2]+R[2][1])/4.0;

                if ((xx > yy) && (xx > zz)) 
                { 
                    if (xx< epsilon) 
                    {
                        r[0] = 0;    r[1] = r[2] = ha; 
                    } else 
                    {
                        double t = sqrt(xx) ;
                        r[0] = double(t * 3.14159265358979323846);
                        r[1] = double(xy/t * 3.14159265358979323846);
                        r[2] = double(xz/t * 3.14159265358979323846);
                    }
                } else if (yy > zz) 
                { 
                    if (yy< epsilon)
                    {
                        r[0] = r[2]  = ha; r[1] = 0;
                    } else
                    {
                        double t = sqrt(yy);
                        r[0] = double(xy/t* 3.14159265358979323846);
                        r[1] = double( t * 3.14159265358979323846);
                        r[2] = double(yz/t* 3.14159265358979323846);
                    }    
                } else 
                {
                    if (zz< epsilon) 
                    {
                        r[0] = r[1] = ha; r[2] = 0;
                    } else
                    {
                        double t  = sqrt(zz);
                        r[0]  = double(xz/ t* 3.14159265358979323846);
                        r[1]  = double(yz/ t* 3.14159265358979323846);
                        r[2]  = double( t * 3.14159265358979323846);
                    }
                }
            }
        }
        else
        {
            a = acos(a);
            double b = 0.5*a/sin(a);
            r[0]    =    double(b*(R[2][1]-R[1][2]));
            r[1]    =    double(b*(R[0][2]-R[2][0]));
            r[2]    =    double(b*(R[1][0]-R[0][1]));
        }
    }

	void setTranslation(const double t[3]) 
    {
        T[0] = t[0];
        T[1] = t[1];
        T[2] = t[2]; 
    }

    Eigen::VectorXd getTranslation()
    {
		Eigen::VectorXd t(3);
        t(0) = T[0];
        t(1) = T[1];
        t(2) = T[2]; 

		return t;
    }

    void getTranslation(double t[3]) const
    {
        t[0] = T[0];
        t[1] = T[1];
        t[2] = T[2]; 

    }

	void setMatrixRotation(const double * r)   
	{   
		for(int i = 0; i < 9; ++i) R[0][i] = r[i];
	}


	void setInvertedRT(const double r[3], const double t[3])
    {
        setRodriguesRotation(r);
        for(int i = 3; i < 9; ++i) R[0][i] = - R[0][i];
        setTranslation(t); T[1] = - T[1]; T[2] = -T[2];
    }    

    void getInvertedRT (double r[3], double t[3]) const
    {
        Camera ci;    ci.setMatrixRotation(R[0]);
        for(int i = 3; i < 9; ++i) ci.R[0][i] = - ci.R[0][i];
        ci.getRodriguesRotation(r);
        getTranslation(t);    t[1] = - t[1]; t[2] = -t[2];
    }

	void setFocalLength(double f)
	{
		this->focalLength = f;
	}

	double getFocalLength()
	{
		return this->focalLength;
	}

	void setProjectionDistortion(double k1, double k2)
	{
		this->k1 = k1;
		this->k2 = k2;
	}

	Eigen::MatrixXd getRotation()
	{
		Eigen::MatrixXd r(3,3);
		for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
		{
			r(i,j) = R[i][j];
		}

		return r;
	}

	void setRotation(Eigen::MatrixXd r)
	{
		R[0][0] = r(0,0);
		R[0][1] = r(0,1);
		R[0][2] = r(0,2);
		R[1][0] = r(1,0);
		R[1][1] = r(1,1);
		R[1][2] = r(1,2);
		R[2][0] = r(2,0);
		R[2][1] = r(2,1);
		R[2][2] = r(2,2);
	}

	void getRotation(double r[3])
	{
		double rr[3];
		getRodriguesRotation(rr);
		r[0] = rr[0];
		r[1] = rr[1];
		r[2] = rr[2];
	}

	double getDistortionk1()
	{
		return this->k1;
	}
	
	double getDistortionk2()
	{
		return this->k2;
	}
	
};

class Projection{
private:
	int cameraIndex;
	int objectPointIndex;
	double x;
	double y;

public:
	double getPointX()
	{
		return x;	
	}

	double getPointY()
	{
		return y;	
	}

	int getCameraIndex()
	{
		return cameraIndex;
	}

	int getObjectPointIndex()
	{
		return objectPointIndex;
	}

	void setObjectPointIndex(int n)
	{
		objectPointIndex = n;
	}

	void setProjectionParameter(int cameraIndex, int objectPointIndex, double x, double y)
	{
		this->x = x;
		this->y = -y; //this is minus here, which cause a big error
		this->cameraIndex = cameraIndex;
		this->objectPointIndex = objectPointIndex;
	}
};

class ObjectPoint{
private:
	double x;
	double y;
	double z;
public:
	void getPoint(double& x, double& y, double& z)
	{
		x = this->x;
		y = this->y;
		z = this->z;	
	}

	void setPoint(double x, double y, double z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	double getPointX()
	{
		return x;
	}
	
	double getPointY()
	{
		return y;
	}
	
	double getPointZ()
	{
		return z;
	}
};
