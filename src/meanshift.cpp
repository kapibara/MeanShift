#include "meanshift.h"

using namespace mean_shift;

std::ostream & operator<<(std::ostream &out , const MatrixRow &row)
{
    for(int i=0; i<row.dim(); i++){
        out << "," << row[i];
    }

    return out;
}

MatrixRow::MatrixRow(const MatrixRow &other):extSource_(other.extSource_)
{
//    std::cerr << "MatrixRow(MatrixRow)" << std::endl;
    dim_ = other.dim_;
    row_ = other.row_;
    if (extSource_){
        //share buffer
        source_ = other.source_;
    }
    else{
        //create own buffer
        source_ = new flann::Matrix<ElemType>(new ElemType[dim_],1,dim_);
        for(int i=0; i< dim_; i++){
            source_->ptr()[i]=other.source_->ptr()[i];
        }
    }
}

/*not optimal -> means are not always good*/
void MeanShift::run()
{
    int seedIdx;
    int countIter;
    int count;
    double meanCount = 0;

    //std::cerr << "before data allocation..." << std::endl;
    //std::cerr.flush();

    int mergeTo;
    MatrixRow mrow(*points_);
    MatrixRow temp(points_->cols);
    MatrixRow mean(points_->cols);
    MatrixRow mean_old(points_->cols);
    //preallocate indices and distances data
    int *indices_data = new int[maxn_];
    flann::L2_3D<ElemType>::ResultType *dist_data = new flann::L2_3D<ElemType>::ResultType[maxn_];
    std::vector<MatrixRow *> perPointVote;
    flann::Matrix<int> indices(indices_data,1,maxn_);
    flann::Matrix<flann::L2_3D<ElemType>::ResultType> dist(dist_data,1,maxn_);
    flann::SearchParams params;
    int ind;
    int neighboursFound;

    //build kd-tree(s)
    if (index_!=0){
        delete index_;
    }
    //std::cerr << "before index created...: " << points_->rows << std::endl;
    //std::cerr.flush();

    index_ = new flann::Index<flann::L2_3D<ElemType> >(*points_, flann::KDTreeSingleIndexParams());
    index_->buildIndex();


    //std::cerr << "index created..." << std::endl;
    //std::cerr.flush();

    //initialize everything else
    //std::cerr << "points_->rows" << points_->rows << std::endl;
    beenVisited_.assign(points_->rows,false);
    data2cluster_.resize(points_->rows);
    free2real_.resize(points_->rows);

    for(int i=0; i< free2real_.size(); i++){
        free2real_[i] = i;
    }
    free_ = points_->rows;

    //don't waste time
    params.checks = -1;
    params.sorted = false;
    clusterCount_ = 0;

    //add the matrix for the first cluster
    perPointVote.push_back(new MatrixRow(points_->rows));

    // std::cerr << "everything is set up..." << std::endl;

    while(free_>0){
        //get next random point;

        //std::cerr << "free: " << free_ << std::endl;

        seedIdx = rand()%free_;
        //seedIdx = 0;
        //find real index
        seedIdx = free2real_[seedIdx];
        //beenVisited_[seedIdx] = true;
        //set mean to the initial point

        //std::cerr << "seedIdx: " << seedIdx << std::endl;

        mean = mrow.setRow(seedIdx);
        //find mode
        countIter = 0;
        //set per-cluster votes count to 0
        perPointVote[clusterCount_]->setTo(0);

 //       std::cerr << "before stating iterations..." << std::endl;

        while(countIter < maxIter_)
        {

            mean_old = mean;
            //find points near the current mean


            neighboursFound = index_->radiusSearch(mean.matrix(),indices,dist,r2_,params);

            //std::cerr << "neighboursFound: " << neighboursFound << std::endl;

            //compute the new weighted mean
            mean.setTo(0);
            meanCount = 0;
            int iterrun = 0;
            for(int i=0; i<indices.cols; i++){
                ind = indices_data[i];
                iterrun++;
                if (ind >= 0){
                    beenVisited_[ind] = true;
                    perPointVote[clusterCount_]->get(ind) +=1;
                    temp = mrow.setRow(ind);
                    if(weights_!=0){
                        temp *= weights_->ptr()[ind];
                        meanCount += weights_->ptr()[ind];
                    }else{
                        meanCount += 1;
                    }
                    mean += temp;

                }else{
                    break;
                }
            }

           mean *= (1.0/((double)meanCount));

           //std::cerr << "mean: " << mean << std::endl;

            //exit cluster search if the center does not move
            if(mean.distL2(mean_old)<r2_*1e-6){
                //std::cerr << "stop by threashold" << std::endl;
                break;
            }

            countIter++;
        }

 //       std::cerr << "iterations finished..." << std::endl;

        //check if we should merge the clusters
        //do it from the end???
        count = 0;
        mergeTo =-1;
        for(std::list<MatrixRow>::iterator i = centers_.begin(); i != centers_.end(); i++ )
        {

            if(mean.distL2((*i)) < r2_/4.0){
                //std::cerr << "*i: " << (*i) << std::endl;
                //std::cerr << "mean.dist: " << mean.distL2((*i)) << " r2/4: " << r2_/4.0<<std::endl;
                //save count
                mergeTo = count;
                //recompute mean
                //std::cerr << "*i" << (*i) << " mean: " << mean << std::endl;

                (*i) += mean;
                (*i) *= 0.5;

                //std::cerr << "*i" << (*i) <<  std::endl;
                //exit loop, dont merge to more then one cluster
                //although it could be possible
                break;
            }

            count++;
        }


        //recompute free_,free2real_ and merge clusters, if needed
        free_ = 0;
        if(mergeTo<0)
        {
            //count the number of free points
            for(int i = 0; i<beenVisited_.size(); i++){
                if (!beenVisited_[i]){
                    free2real_[free_] = i;
                    free_++;
                }
            }
            //add new center
            centers_.push_back(mean);
            //add new cluster
            clusterCount_++;
            perPointVote.push_back(new MatrixRow(points_->rows));
        }else{

            //std::cerr << "merging to " << mergeTo << std::endl;
            for(int i = 0; i<beenVisited_.size(); i++){
                //aggregate per-point votes
                perPointVote[mergeTo]->get(i) += perPointVote[clusterCount_]->get(i);
                //count the number of free points
                if (!beenVisited_[i]){
                        free2real_[free_] = i;
                        free_++;
                }
            }
            //don't add a new center
        }


    }

    int maxind = 0;
    clusterSizes_.assign(clusterCount_,0);
    //final step - aggregate points affiliation
    for(int i=0; i<data2cluster_.size();i++){
        maxind = 0;
        for(int j=1; j<clusterCount_; j++){
            if (perPointVote[j]->get(i) > perPointVote[maxind]->get(i)){
                maxind = j;
            }
        }

        //assign the cluster
        data2cluster_[i] = maxind;
        //increase cluster size by 1
        clusterSizes_[maxind]++;
    }


    if(recomputeCenters_){
        std::vector<MatrixRow> newcenters;
        std::vector<double> count;
        MatrixRow tmp(points_->cols);
        tmp.setTo(0);

        newcenters.assign(clusterCount_,tmp);
        count.assign(clusterCount_,0);

        //aggregate
        for(int i=0; i<data2cluster_.size();i++){
            temp = mrow.setRow(i);
            if(weights_!=0){
                temp*=weights_->ptr()[i];
                count[data2cluster_[i]]+=weights_->ptr()[i];
            }else{
                count[data2cluster_[i]]++;
            }
            newcenters[data2cluster_[i]]+=temp;
        }
        //normalize
        //copy to centers
        std::list<MatrixRow>::iterator itor = centers_.begin();
        for(int i=0; i< clusterCount_; i++){
            if(count[i]>0){
                //there are still some votes from this class
                newcenters[i]*=(1.0/count[i]);
                //std::cerr << "old center: " << (*itor) << std::endl;
                //std::cerr << "new center: " << newcenters[i] << std::endl;
                (*itor) = newcenters[i];
            }
            itor++;
        }


    }

    //clear perPointVote
    for(int i=0; i<perPointVote.size(); i++){
        //delete MatrixRow
        delete perPointVote[i];
    }
    //clean allocated indices data
    delete [] indices_data;
    delete [] dist_data;
}
