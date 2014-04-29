#ifndef MEANSHIFT_H
#define MEANSHIFT_H

#include <cstdlib>
#include <list>

#include "flann/flann.hpp"

namespace mean_shift{
class MatrixRow;
};

std::ostream & operator<<(std::ostream &out , const mean_shift::MatrixRow &row);


namespace mean_shift{


typedef double ElemType;
typedef unsigned int IndexType;

class MatrixRow
{
public:

    //row based on a source
    MatrixRow(flann::Matrix<ElemType> &source):extSource_(true)
    {
        source_ = &source;
        dim_ = source.cols;
        row_ = 0;
    }

    MatrixRow(const MatrixRow &other);

    //row without external source
    MatrixRow(int dim):extSource_(false)
    {
        dim_ = dim;
        source_ = new flann::Matrix<ElemType>(new ElemType[dim],1,dim);
        row_ = 0;
    }

    ~MatrixRow()
    {

        if(!extSource_){
            delete [] source_->ptr();
        }
    }

    int dim() const{
        return dim_;
    }


    void setTo(ElemType val)
    {
        ElemType *row = source_->operator [](row_);

        for(int i=0; i< dim_; i++){
            row[i]=val;
        }


    }

    //set current row;
    MatrixRow & setRow(IndexType row)
    {
        row_ = row;
        return (*this);
    }

    flann::Matrix<ElemType> matrix()
    {
        if (extSource_){
            return flann::Matrix<ElemType>(source_->operator [](row_),1,dim_);
        }else{
            return (*source_);
        }

    }

    ElemType &get(int col)
    {
        return (source_->operator [](row_))[col];
    }

    ElemType &operator[](int col)
    {
        return (source_->operator [](row_))[col];
    }

    ElemType operator[](int col) const
    {
        return (source_->operator [](row_))[col];
    }

    MatrixRow &operator=(const MatrixRow &other)
    {
        ElemType *row = source_->operator [](row_);
        ElemType *otherrow = other.source_->operator [](other.row_);

        for(int i=0; i< dim_; i++){
            row[i]=otherrow[i];
        }

        return (*this);
    }

    MatrixRow &operator+=(const MatrixRow &other)
    {
        //omit dimensionality check
        ElemType *row = source_->operator [](row_);
        ElemType *otherrow = other.source_->operator [](other.row_);
        for(int i=0; i< dim_; i++){
            row[i]+=otherrow[i];
        }

        return (*this);
    }

    MatrixRow &operator*=(ElemType alpha)
    {
        //omit dimensionality check
        ElemType *row = source_->operator [](row_);
        for(int i=0; i< dim_; i++){
            row[i]*=alpha;
        }

        return (*this);
    }

    ElemType distL2(const MatrixRow &other)
    {
        //omit dimensionality check
        ElemType *row = source_->operator [](row_);
        ElemType *otherrow = other.source_->operator [](other.row_);
        ElemType result = 0;
        ElemType diff;
        for(int i=0; i< dim_; i++){
            diff = row[i]-otherrow[i];
            result+=diff*diff;
        }

        return result;
    }

    ElemType *ptr()
    {
        return source_->ptr();
    }

private:
    flann::Matrix<ElemType> *source_;

    int dim_;
    IndexType row_;
    const bool extSource_;
};

class MeanShift
{
public:
    MeanShift()
    {
        points_ = 0;
        r_ = 100;
        maxIter_ = 10;
        r2_ = r_*r_;
        index_ = 0;
        weights_ = 0;
        maxn_ = 5000;
        recomputeCenters_=true;
    }

    ~MeanShift()
    {
        delete index_;
    }

    void setRecomputeCenters(bool re){
        recomputeCenters_ = re;
    }

    void setRadius(double r)
    {
        r_ = r;
        r2_ = r_*r_;
    }

    void setMaxNeigboursCount(IndexType maxn)
    {
        maxn_ = maxn;
    }

    void setMaxIter(int maxIter)
    {
        maxIter_ = maxIter;
    }

    void setPoints(flann::Matrix<ElemType> &points)
    {
        Clean();
        //set points
        points_ = &points;
        //unset weights
        weights_ = 0;

    }

    void setPoints(flann::Matrix<ElemType> &points, flann::Matrix<ElemType> &weights)
    {
        Clean();
        //set points
        points_ = &points;
        //set weights
        weights_ = &weights;
    }

    IndexType getClusterNumber()
    {
        return clusterCount_;
    }

    //preallocated calls
    void getClusterIndices(std::vector<IndexType> &data2cluster)
    {
        for(int i=0; i<data2cluster_.size(); i++){
            data2cluster[i] = data2cluster_[i];
        }
    }

    //preallocated calls
    void getClusterCenters(flann::Matrix<ElemType> &centers)
    {
        MatrixRow wrapper(centers);
        int rowcount = 0;

        for(std::list<MatrixRow>::iterator i = centers_.begin(); i != centers_.end(); i++ )
        {
            wrapper.setRow(rowcount) = (*i);
            rowcount++;
        }

    }

    void getClusterSizes(std::vector<IndexType> &sizes)
    {
        for(int i = 0; i < clusterCount_; i++ )
        {
            sizes[i] = clusterSizes_[i];
        }

    }

    //since there is no copy constructor, lets cheat
    flann::Matrix<IndexType> getClusterIndices()
    {
        IndexType *data = new IndexType[points_->rows];
        flann::Matrix<IndexType> result(data,points_->rows,1);

        for(int i=0; i< points_->rows; i++){
            data[i] = data2cluster_[i];
        }

        return result;
    }

    void getClusterCenter(int centerInd, std::vector<ElemType> &elem)
    {
        int ind = 0;
        std::list<MatrixRow>::iterator i = centers_.begin();
        while(ind < centerInd)
        {
            ind++;
            i++;
        }

        std::copy((*i).ptr(),((*i).ptr()+(*i).dim()),elem.begin());
    }

    //since there is no copy constructor, lets cheat
    flann::Matrix<ElemType> getClusterCenters()
    {
        ElemType *data = new ElemType[clusterCount_*points_->cols];
        flann::Matrix<ElemType> result(data,clusterCount_,points_->cols);
        MatrixRow wrapper(result);
        int rowcount = 0;

        for(std::list<MatrixRow>::iterator i = centers_.begin(); i != centers_.end(); i++ )
        {
//            std::cerr << "rows(getClCent): " << (*i) << std::endl;
            wrapper.setRow(rowcount) = (*i);
//            std::cerr << "wrapper(getClCent): " << wrapper << std::endl;
            rowcount++;
        }

        return result;
    }

    //since there is no copy constructor, lets cheat
    flann::Matrix<IndexType> getClusterSizes()
    {
        flann::Matrix<IndexType> result(new IndexType[clusterCount_],1,clusterCount_);

        for(int i = 0; i < clusterCount_; i++ )
        {
            (result.ptr())[i] = clusterSizes_[i];
        }

        return result;

    }

    void Clean()
    {
        centers_.clear();
        clusterCount_ = 0;
        points_ = 0;
        weights_ = 0;
    }

    void run();

private:

    //options
    ElemType r_;
    ElemType r2_;
    int maxIter_;

    bool recomputeCenters_;
    IndexType maxn_;
    flann::Matrix<ElemType> *points_;
    flann::Matrix<ElemType> *weights_;
    flann::Index<flann::L2_3D<ElemType> > *index_;

    std::vector<bool> beenVisited_;
    std::vector<IndexType> data2cluster_;
    std::vector<IndexType> free2real_;
    IndexType free_;
    std::list<MatrixRow> centers_;
    std::vector<IndexType> clusterSizes_;
    IndexType clusterCount_;

};

};

#endif // MEANSHIFT_H
