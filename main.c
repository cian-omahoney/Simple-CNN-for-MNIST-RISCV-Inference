
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


char mnistDirectory[] = "../mnist/";
char sz_Trndata[] = "train-images-idx3-ubyte";
char sz_Trnlabels[] = "train-labels-idx1-ubyte";
char sz_Tstdata[] = "t10k-images-idx3-ubyte";
char sz_Tstlabels[] = "t10k-labels-idx1-ubyte";

int nPass;
float xf3;
double fbb;
int vpos[65536];

int n_images;
int n_trn_images;
int n_tst_images;
unsigned char * pb_trn_data;
unsigned char * pb_tst_data;
int *pi_trn_labels;
int *pi_tst_labels;

unsigned char mrfv[784];




int * read_mnist_labels(char *szfname)
{
	int i;
	int *ptr = 0;
	unsigned char *pcha;
	FILE *fi = fopen(szfname,"rb");
	if(fi)
	{
		int ns = fseek(fi,0,SEEK_END);
		if(ns==0)	//success
		{
			long ln = ftell(fi);
			int nSize = (int) ln - 8;
			if(nSize>0)
			{
				n_images = nSize;
				ptr = (int *) malloc(n_images*sizeof(int));
				pcha = (unsigned char *) malloc(n_images);
				fseek(fi,8,SEEK_SET);
				fread((void *) pcha,1,n_images,fi);
				for(i=0;i<n_images;i++)
				{
					ptr[i] = (int) pcha[i];
				}
				free(pcha);
			}
		}

		fclose(fi);
	}

	return ptr;
}



unsigned char * read_mnist_data(char *szfname)
{
	unsigned char *pcha = 0;

	FILE *fi = fopen(szfname,"rb");
	if(fi)
	{
		int ns = fseek(fi,0,SEEK_END);
		if(ns==0)	//success
		{
			long ln = ftell(fi);
			int nSize = (int) ln - 16;
			if(nSize>0)
			{
				int n_pix = nSize;
				int n_pict = n_pix/784;
				if(n_pict==n_images)
				{
					pcha = (unsigned char *) malloc(n_pix);

					fseek(fi,16,SEEK_SET);
					fread((void *) pcha,1,n_pix,fi);

				}
			}
		}
	}

	return pcha;
}


//-------------------------------------------------------------
void shuffle_training_set()
{
	int i,ix,vp;
	
	for(i=0;i<60000;i++) vpos[i]=i;
	
	for(i=0;i<59999;i++)
	{
		ix = i + rand() / (RAND_MAX / (59999 - i) + 1);
		vp = vpos[ix];
		vpos[ix] = vpos[i];
		vpos[i] = vp;

	}
}



int read_mnist(void)
{
    char bbuf[512];
    int n = 0;

    sprintf(bbuf,"%s%s",mnistDirectory,sz_Trnlabels);
    pi_trn_labels = read_mnist_labels(bbuf);

    n_trn_images = n_images;

    sprintf(bbuf,"%s%s",mnistDirectory,sz_Trndata);
    pb_trn_data = read_mnist_data(bbuf);

    sprintf(bbuf,"%s%s",mnistDirectory,sz_Tstlabels);
    pi_tst_labels = read_mnist_labels(bbuf);

    n_tst_images = n_images;

    sprintf(bbuf,"%s%s",mnistDirectory,sz_Tstdata);
    pb_tst_data = read_mnist_data(bbuf);

    if(pi_trn_labels!=0 && pb_trn_data!=0 && pi_tst_labels!=0 && pb_tst_data!=0)
    {
        n = 1;
        printf("Data read OK\n");
    }
    else
    {
        printf("Oops. Data read failed.\n");
    }

	return n;
}



//-------------------------------------------------------------
float dist_frand(void)
{
    int irx,idr;
    float fx,fd,fr;

	irx = rand();
	fx = (float) irx;
	idr = RAND_MAX/2;
	fd = (float) idr;
	fr = (fx-fd)/fd;

	return fr;
}

//-------------------------------------------------------------
float dist_frandp(void)
{
    int irx,idr;
    float fx,fd,fr;

	irx = rand();
	fx = (float) irx;
	idr = RAND_MAX;
	fd = (float) idr;
	fr = fx/fd;

	return fr;
}



//-------------------------------------------------------------
void distort_imageb(unsigned char *pch)
{
    int i,j,i1,j1,idsx,idsy,npos;
    float xtr,xtl,xbl,xbr,ytr,ytl,ybl,ybr;
    float dtx,dbx,dlhx,ddx,vx,dxv,xlv;
    float dty,dby,dlhy,ddy,vy,dyv,ylv;
    float fm = (float) 27.0;
    float fh = (float) 0.5;
    float f13 = (float) 13.5;

	xtl = xf3*dist_frand();
	xtr = xf3*dist_frand();
	xbl = xf3*dist_frand();
	xbr = xf3*dist_frand();

	ytl = xf3*dist_frand();
	ytr = xf3*dist_frand();
	ybl = xf3*dist_frand();
	ybr = xf3*dist_frand();


	dtx = (xtr-xtl)/fm;
	dbx = (xbr-xbl)/fm;
	dlhx = (xbl-xtl)/fm;
	ddx = (dbx-dtx)/fm;
	dty = (ytr-ytl)/fm;
	dby = (ybr-ybl)/fm;
	dlhy = (ybl-ytl)/fm;
	ddy = (dby-dty)/fm;

	xlv = xtl+fh;
	dxv = dtx;
	ylv = ytl+fh;
	dyv = dty;
	npos=0;
	for(i=0;i<28;i++) 
	{
		vx = xlv;
		vy = ylv;
		for(j=0;j<28;j++)
		{
			idsx = (int) vx;
			idsy = (int) vy;

			i1=i+idsy;
			j1=j+idsx;

			if(i1<0) i1=0;
			if(i1>27) i1=27;
			if(j1<0) j1=0;
			if(j1>27) j1=27;

			mrfv[npos++]=*(pch+i1*28+j1);

			vx+=dxv;
			vy+=dyv;
		}
		xlv+=dlhx;
		dxv+=ddx;
		ylv+=dlhy;
		dyv+=ddy;
	}

	for(i=0;i<784;i++) *(pch+i)=mrfv[i]; 
}



void empty_arrays(void)
{
	if(pb_tst_data)
	{
		free(pb_tst_data);
	}
	if(pb_trn_data)
	{
		free(pb_trn_data);
	}
	if(pi_trn_labels)
	{
		free(pi_trn_labels);
	}
	if(pi_tst_labels)
	{
		free(pi_tst_labels);
	}
}

extern double lambda,meanerr,maxerr,fthre;
extern int nflerr;
extern unsigned char chrf[784];


void cnn_init(void);
int cnn_Recogn(void);
void cnn_print_network_info(FILE *fp);
int cnn_LearnTo(int ndxtarg);
void cnn_reset(void);
void cnn_back(void);
void cnn_SerialiseParams(void);




void do_experiment()
{
    char msgbuf[256];
    time_t ltime;
    int numerrtrn,numupd,numerrtst,ntrnpat,npos;
    int i,j,k,l;
    int nrec;
    int staterr[10];
    FILE *fo;

    unsigned char *pbtrn, *pbtst;
    double dlth,y;


	nPass = 1; //50
	lambda = 0.01; //0.01

	srand(12345);
	cnn_init();
	cnn_reset();
	fthre = 0.6;
	dlth = (fthre - 0.3) / ((double)nPass);

	y = log(0.05)/((double)nPass);
	fbb = exp(y);


	ntrnpat = 60000;

	xf3 = (float) 5.0;


	fo=fopen("mnist_demo_log.txt","a+");

	cnn_print_network_info(fo);

	time( &ltime );
	sprintf(msgbuf, "Learning start: %s\n",ctime( &ltime ) );
	printf(msgbuf);
	fprintf(fo,"%s",msgbuf);

	printf("Epoch\tUpdates\tTrn_err\tTest_err\tLambda\n");
	fprintf(fo,"Epoch\tUpdates\tTrn_err\tTest_err\tLambda\n");
	fclose(fo);


	for(i=0;i<nPass;i++)
	{
		numerrtrn = 0;
		numupd = 0;
		numerrtst = 0;
		shuffle_training_set();

		for(j=0;j<ntrnpat;j++)
		{
			npos = vpos[j];
			pbtrn = pb_trn_data + 784*npos;
			for(l=0;l<784;l++) chrf[l] = *(pbtrn + l);
			distort_imageb(chrf);

			numupd+=cnn_LearnTo(pi_trn_labels[npos]);
			numerrtrn+=nflerr;
		}


		for(j=0;j<n_tst_images;j++)
		{
			pbtst = pb_tst_data + j*784;
			for(l=0;l<784;l++) chrf[l] = *(pbtst + l);

			nrec = cnn_Recogn();
			if(nrec!=pi_tst_labels[j])
			{
				++numerrtst;
			}
		}

		sprintf(msgbuf,"%d\t%d\t%d\t%d\t\t%f\n",i,numupd,numerrtrn,numerrtst,lambda);
		printf(msgbuf);
		fo=fopen("mnist_demo_log.txt","a+");
		fprintf(fo,"%s",msgbuf);
		fclose(fo);

		lambda*=fbb;
		fthre -= dlth;
	}

    cnn_SerialiseParams();
    cnn_DeserialiseParams();

	for(j=0;j<10;j++) staterr[j]=0;

	fo=fopen("mnist_demo_log.txt","a+");
	printf("Recognition\n");
	fprintf(fo,"Recognition\n");
	numerrtst = 0;
	for(j=0;j<n_tst_images;j++)
	{
		pbtst = pb_tst_data + j*784;
		for(l=0;l<784;l++) chrf[l] = *(pbtst + l);

		nrec = cnn_Recogn();
		if(nrec!=pi_tst_labels[j])
		{
			++numerrtst;
			k=pi_tst_labels[j];
			++staterr[k];
		}
	}
		
		
	time( &ltime );
	printf( "The end: %s\n", ctime( &ltime ) );
	fprintf(fo, "The end: %s\n", ctime( &ltime ) );
	printf("Test_set:%d errors\n",numerrtst);
	fprintf(fo,"Test_set:%d errors\n",numerrtst);
	for(j=0;j<10;j++)
	{
		fprintf(fo,"%d: %d errors\n",j,staterr[j]);
	}


	fprintf(fo,"-----------------------------------------------------------\n\n\n");

	fclose(fo);
}



int main(int argc, char* argv[])
{
	int ndata_all_ok;

	ndata_all_ok = read_mnist();

	if(ndata_all_ok == 1)
	{
		do_experiment();
	}
	else
	{
		printf("Mission failed.\n");
	}

	empty_arrays();

	return 0;
}
