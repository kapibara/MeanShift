
#include "meanshift.h"

#include "flann/io/hdf5.h"

using namespace mean_shift;

int main(int argc, char **argv)
{
    float *data = new float[2*3];
    data[0] = data[1] = data [2] = 1;
    data[3] = data[4] = data [5] = 1;


    flann::Matrix<ElemType> dataset1;
    flann::Matrix<ElemType> dataset2;
    flann::Matrix<ElemType> centers(0,0,0);
    flann::Matrix<IndexType> clids(0,0,0);
    flann::Matrix<IndexType> clsizes(0,0,0);

    {

        MeanShift mshift;

        mshift.setRadius(100);
        mshift.setMaxIter(10);
        mshift.setMaxNeigboursCount(30000);

        flann::load_from_file(dataset2,"votes.hdf5","/test");

        std::cerr << "dataset loaded:" << dataset2.rows << std::endl;

        mshift.setPoints(dataset2);

        std::cerr << "points set" << std::endl;

        mshift.run();

        std::cerr << "meanshift finished: " << mshift.getClusterNumber() << std::endl;
        centers = mshift.getClusterCenters();
        clids = mshift.getClusterIndices();
        clsizes = mshift.getClusterSizes();

        try{
            flann::save_to_file(centers,"result.hdf5","/ce");
            flann::save_to_file(clids,"result.hdf5","/ind");
            flann::save_to_file(clsizes,"result.hdf5","/s");
            flann::save_to_file(dataset2,"result.hdf5","/v");
        }catch(flann::FLANNException &e){
            std::cerr << "exception: " << e.what() << std::endl;
        }
/*
        flann::load_from_file(dataset1,"test.hdf5","/votes1019");

        std::cerr << "dataset loaded:" << dataset1.rows << std::endl;

        mshift.setPoints(dataset1);

        std::cerr << "points set (1019)" << std::endl;

        mshift.run();

        std::cerr << "meanshift finished (1019): "<< mshift.getClusterNumber() << std::endl;

        centers = mshift.getClusterCenters();
        clids = mshift.getClusterIndices();
        clsizes = mshift.getClusterSizes();

        try{
            flann::save_to_file(centers,"result.hdf5","/centers1019");
            flann::save_to_file(clids,"result.hdf5","/indices1019");
            flann::save_to_file(clsizes,"result.hdf5","/sizes1019");
        }catch(flann::FLANNException &e){
            std::cerr << "exception: " << e.what() << std::endl;
        }
*/
    }



    return 0;
}
