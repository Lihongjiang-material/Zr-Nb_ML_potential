// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the kernel ridge regression 
    object from the dlib C++ Library.

    This example will train on data from the sinc function.

*/

#include <iostream>
#include <vector>

#include <dlib/svm.h>

using namespace std;
using namespace dlib;


int main(int argc,char *argv[])
{
    // Here we declare that our samples will be 1 dimensional column vectors.  
    typedef matrix<double,132,1> sample_type;
    int ncolm = 132;
    int nfeature = 132;
	
    // Now sample some points from the sinc() function
    sample_type m,fvect_dev;
    std::vector<sample_type> samples,sample_Etest,sample_Ftest,sample_Ptest;
    std::vector<double> labels,label_Etest,label_Ftest,label_Ptest;

    std::ifstream fin;
    std::ofstream fout;
	
    fin.open("Train_Data.txt");                   
    if(fin.fail())
    {
       cout<<"Input file opening failed.\n";
       exit(1);
    }

   double target,mt;

    while(!fin.eof())
      {
           for(int j=0;j<ncolm;j++)
	      if(j<nfeature)
               fin>>m(j);
	      else 
		fin>>mt;
            fin>>target;
            samples.push_back(m);
            labels.push_back(target);
        } 
	fin.close();

    fin.open("TrainData.txt");	
    while(!fin.eof())
      {
           for(int j=0;j<ncolm;j++)
	     if(j<nfeature)
               fin>>m(j);
             else 
	       fin>>mt;
            fin>>target;
            sample_Etest.push_back(m);
            label_Etest.push_back(target);
        } 
	fin.close();	

/*    fin.open("T_pv3.txt");
    while(!fin.eof())
      {
           for(int j=0;j<ncolm;j++)
             if(j<nfeature)
               fin>>m(j);
             else
               fin>>mt;
            fin>>target;
            sample_Ptest.push_back(m);
            label_Ptest.push_back(target);
        }
        fin.close();

    fin.open("TestFData.txt");	
    while(!fin.eof())
      {
           for(int j=0;j<ncolm;j++)
	     if(j<nfeature)
               fin>>m(j);
             else 
	       fin>>mt;
            fin>>target;
            sample_Ftest.push_back(m);
            label_Ftest.push_back(target);
        } 
	fin.close();
    */	

    vector_normalizer<sample_type> normalizer;
    normalizer.train(samples);

   //cout<<normalizer.means()<<endl;
   //cout<<normalizer.std_devs()<<endl;
   if(atoi(argv[1])==1){
   for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]);

    for (unsigned long i = 0; i < sample_Etest.size(); ++i)
        sample_Etest[i] = normalizer(sample_Etest[i]);

/*    for (unsigned long i = 0; i < sample_Ptest.size(); ++i)
        sample_Ptest[i] = normalizer(sample_Ptest[i]);

    for (unsigned long i = 0; i < sample_Ftest.size(); ++i)
        sample_Ftest[i] = normalizer(sample_Ftest[i]);*/
   }
 
    randomize_samples(samples, labels);

    typedef linear_kernel<sample_type> kernel_type;

    cout << "\ndoing a grid cross-validation" << endl;
    matrix<double> params = logspace(log10(1e-6),log10(1e0),7);
     
    matrix<double> best_result(2,1);
    best_result = 0;
    double best_lambda = 1e-6;
/*    for(long col =0 ;col <params.nc(); ++col)
    {
        // tell the trainer the parameters we want to use
          const double lambda = params(0,col);


        krr_trainer<kernel_type> trainer_cv;
        trainer_cv.set_lambda(lambda);

       matrix<double> result = cross_validate_regression_trainer(trainer_cv, samples, labels,5);

       if(sum(result)> sum(best_result))
          {
             best_result = result;
             best_lambda = lambda;
              }
    }	
	
    cout <<"\n best result of grid search: " <<sum(best_result) <<endl;
  */  cout <<"  best lambda: "<<best_lambda<<endl;
	
   rr_trainer<kernel_type> trainer;
    trainer.set_lambda(best_lambda);

    // now train a function based on our sample points
    decision_function<kernel_type> test = trainer.train(samples, labels);


    cout <<"b0="<<test.b<<endl;
   //calculate the derivate of kernel   
   for(int j=0;j<nfeature;j++)
       fvect_dev(j) = 0.0;

    for(int i=0;i<test.basis_vectors.nr();i++)
         fvect_dev += test.alpha(i)*test.basis_vectors(i);


     double inter_p = test.b;

    if(atoi(argv[1])==1){
     for(int j=0;j<nfeature;j++){
        fvect_dev(j) = fvect_dev(j)*normalizer.std_devs()(j);
      inter_p += normalizer.means()(j)*fvect_dev(j);
    }
 //    inter_p -= fvect_dev(nfeature);
     cout<<"interp: "<<inter_p<<endl;
    }

    FILE * fp= fopen("Param_ML_pot.txt","w");
    fprintf(fp,"# Fitted ML parameters\n");
    fprintf(fp,"#2 Zr Nb 91.224 92.91\n");
    fprintf(fp,"%d %d\n",nfeature,1);
    for(int j=0;j<nfeature;j++)
      fprintf(fp,"%lg\n",fvect_dev(j));
    fprintf(fp,"%lg\n",inter_p);
    fclose(fp);



    double fmean = 0.0;
	double fdev = 0.0;
	std::vector<double> loo_Pvalues,loo_Evalues,loo_Fvalues,loo_values; 
	
	fout.open("rst.txt");
	for(int i = 0;i<sample_Etest.size();i++)
	{	
	   m = sample_Etest[i];
	   double y_kr = test(m);
	   loo_Evalues.push_back(y_kr);
	   
	   fout<<i+1<<" "<<label_Etest[i]<<" "<<y_kr<<endl;
	   
	   fmean += fabs(label_Etest[i]);
	   fdev  +=fabs(label_Etest[i]-y_kr);
	}
	fout.close();

   printf("score=%lg\n",r_squared(label_Etest,loo_Evalues));
   printf("rms=%.6f,prf=%.6f\n",fdev/sample_Etest.size(),fdev/fmean);	


/*  fmean = 0.0;
  fdev = 0.0;

        for(int i = 0;i<sample_Ptest.size();i++)
        {
           m = sample_Ptest[i];
           double y_kr = test(m);
           loo_Pvalues.push_back(y_kr);

           fmean += fabs(label_Ptest[i]);
           fdev  +=fabs(label_Ptest[i]-y_kr);
        }

   printf("score=%lg\n",r_squared(label_Ptest,loo_Pvalues));
   printf("rms=%.6f,prf=%.6f\n",fdev/sample_Ptest.size(),fdev/fmean);

  fmean = 0.0;
  fdev = 0.0; 
 for(int i = 0;i<sample_Ftest.size();i++)
	{	
	   m = sample_Ftest[i];
	   double y_kr = 0.0;

           for(int j=0;j<nfeature;j++)
                y_kr += m(j)*fvect_dev(j);


     loo_Fvalues.push_back(y_kr);
	   
	   fmean += fabs(label_Ftest[i]);
	   fdev  +=fabs(label_Ftest[i]-y_kr);
	}

   printf("score=%lg\n",r_squared(label_Ftest,loo_Fvalues));
   printf("rms=%.6f,prf=%.6f\n",fdev/sample_Ftest.size(),fdev/fmean); 
*/

  fmean = 0.0;
  fdev = 0.0;
 for(int i = 0;i<samples.size();i++)
        {
           m = samples[i];
           double y_kr = test(m);
           loo_values.push_back(y_kr);

           fmean += fabs(labels[i]);
           fdev  +=fabs(labels[i]-y_kr);

        }

   printf("score=%lg\n",r_squared(labels,loo_values));
   printf("rms=%.6f,prf=%.6f\n",fdev/samples.size(),fdev/fmean);

}


