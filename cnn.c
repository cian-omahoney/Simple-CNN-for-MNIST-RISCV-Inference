#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define RFW			6
#define INPW		28
#define L1W			12
#define L2W			4
#define NN_LAYER1	12
#define NN_LAYER2	32
#define NN_OUTPUT	10
#define INPV_LEN	784
#define TBL_LEN		256

double inpf[INPW][INPW];
double cno1[NN_LAYER1][L1W][L1W];
double cno2[NN_LAYER2][L2W][L2W];
unsigned char chrf[INPV_LEN];

double e2[NN_LAYER2][L2W][L2W]; 
double e1[NN_LAYER1][L1W][L1W];
double l3[NN_OUTPUT];
double t3[NN_OUTPUT];
double e3[NN_OUTPUT];

double lambda,fthre;
int nflerr;


double cw1[NN_LAYER1][RFW][RFW];
double cw2[NN_LAYER2][NN_LAYER1][RFW][RFW];
double w3[NN_OUTPUT][NN_LAYER2][L2W][L2W];

double dgr[TBL_LEN];

double thr1[NN_LAYER1];
double thr2[NN_LAYER2];
double thro[NN_OUTPUT];


//--------------------------------------------------
void cnn_init(void)
{
	int i;
	double fv;
	double kz = 0.00390625;

	for(i=0;i<TBL_LEN;i++)
	{
		fv = (double) i;
		dgr[i] = kz*fv;
	}
}


//-------------------------------------------------
int cnn_Recogn(void)
{
	int iret=0;
	double fsum,z;
	int i,j,n,nn,cx,cy,rx,ry;
	double sq[NN_OUTPUT];
	double se[NN_OUTPUT];
	double qmax = 10000000.0;
	unsigned char chx;


	for(i=0;i<INPW;i++) for(j=0;j<INPW;j++)
	{
		chx = chrf[i*INPW+j];
		nn = (int) chx;
		inpf[i][j] = dgr[nn];
	}
	
	for(n=0;n<NN_LAYER1;n++) for(cy=0;cy<L1W;cy++) for(cx=0;cx<L1W;cx++)
	{
		fsum=thr1[n];
		for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
		{
			fsum+=cw1[n][ry][rx]*inpf[2*cy+ry][2*cx+rx];
		}
		if(fsum>0) z = fsum;
		else	   z = 0;	
		cno1[n][cy][cx] = z;
	}


	for(n=0;n<NN_LAYER2;n++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
	{
		fsum=thr2[n];
		for(nn=0;nn<NN_LAYER1;nn++) for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
		{
			fsum+=cw2[n][nn][ry][rx]*cno1[nn][2*cy+ry][2*cx+rx];
		}
		if(fsum>0) z = fsum;
		else	   z = 0;	
		cno2[n][cy][cx] = z;
	}


	for(n=0;n<NN_OUTPUT;n++)
	{
		fsum=thro[n];
		for(nn=0;nn<NN_LAYER2;nn++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
		{
			fsum+=w3[n][nn][cy][cx]*cno2[nn][cy][cx];
		}
		sq[n]=fsum;
	}

	z=-qmax;
	for(n=0;n<NN_OUTPUT;n++)
	{
		if(z<sq[n])
		{
			z=sq[n];
		}
	}

	for(n=0;n<NN_OUTPUT;n++) sq[n]-=z;
	for(n=0;n<NN_OUTPUT;n++) se[n]=exp(sq[n]);

	fsum=0;
	for(n=0;n<NN_OUTPUT;n++) fsum+=se[n];
	for(n=0;n<NN_OUTPUT;n++) l3[n]=se[n]/fsum;


	z=-1.0;
	for(n=0;n<NN_OUTPUT;n++)
	{
		if(l3[n]>z)
		{
			z=l3[n];
			iret=n;
		}
	}

	return iret;
}

//-------------------------------------------------
void cnn_print_network_info(FILE *fp)
{
	int n1,n2,n3,nn;

	n1 = NN_LAYER1+NN_LAYER1*RFW*RFW;
	n2 = NN_LAYER2+NN_LAYER2*NN_LAYER1*RFW*RFW;
	n3 = NN_OUTPUT+NN_OUTPUT*NN_LAYER2*L2W*L2W;
	nn = n1+n2+n3;

	if(fp)
	{
		fprintf(fp,"NN_LAYER1:\t%d\nNN_LAYER2:\t%d\nNN_OUTPUT:\t%d\nn1=%d\nn2=%d\nn3=%d\n ALL: %d\n",NN_LAYER1,NN_LAYER2,NN_OUTPUT,n1,n2,n3,nn);
	}
}



//----------------------------------------------------------------------------------------
double cnn_frand1()
{
	int irx = rand();
	double fx = (double) irx;
	int idr = RAND_MAX/2;
	double fd = (double) idr;
	double fr = (fx-fd)/fd;
	return fr;
}
//----------------------------------------------------------------------------------------
double cnn_frand2()
{
	int irx = rand();
	double fx = (double) irx;
	int idr = RAND_MAX;
	double fd = (double) idr;
	double fr = fx/fd;
	return fr;
}
//----------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------
void cnn_reset(void)
{
	int n1,n2,n3,cx,cy,rx,ry;
	double fs = 0.04;


	for(n1=0;n1<NN_LAYER1;n1++) for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
	{
		double fx = cnn_frand1();
		cw1[n1][ry][rx] = fs*fx; 
	}

	for(n2=0;n2<NN_LAYER2;n2++) for(n1=0;n1<NN_LAYER1;n1++) for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
	{
		double fx = cnn_frand1();
		cw2[n2][n1][ry][rx] = fs*fx;
	}

	for(n3=0;n3<NN_OUTPUT;n3++) for(n2=0;n2<NN_LAYER2;n2++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
	{
		double fx = cnn_frand1();
		w3[n3][n2][cy][cx] = fs*fx;
	}


	for(n1=0;n1<NN_LAYER1;n1++) thr1[n1]=0;
	for(n2=0;n2<NN_LAYER2;n2++) thr2[n2]=0;
	for(n3=0;n3<NN_OUTPUT;n3++) thro[n3]=0;
}


//----------------------------------------------------------------------------------------
void back(void)
{
	int cx,cy,rx,ry,n1,n2,n4;
	double fsum,z,lz;


	for(n2=0;n2<NN_LAYER2;n2++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
	{
		z = cno2[n2][cy][cx];
		if(z>0)
		{
			fsum=0;
			for(n4=0;n4<NN_OUTPUT;n4++)
			{
				fsum+=w3[n4][n2][cy][cx]*e3[n4];
			}
			e2[n2][cy][cx] = fsum;
		}
		else
		{
			e2[n2][cy][cx] = 0;
		}
	}


	for(n1=0;n1<NN_LAYER1;n1++) for(cy=0;cy<L1W;cy++) for(cx=0;cx<L1W;cx++)
		e1[n1][cy][cx]=0;

	for(n2=0;n2<NN_LAYER2;n2++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
	{
		for(n1=0;n1<NN_LAYER1;n1++) for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
		{
			e1[n1][2*cy+ry][2*cx+rx]+=cw2[n2][n1][ry][rx]*e2[n2][cy][cx];
		}
	}

	for(n1=0;n1<NN_LAYER1;n1++) for(cy=0;cy<L1W;cy++) for(cx=0;cx<L1W;cx++)
	{
		z = cno1[n1][cy][cx];
		if(z==0)
		{
			e1[n1][cy][cx] = 0;
		}
	}

	for(n4=0;n4<NN_OUTPUT;n4++) for(n2=0;n2<NN_LAYER2;n2++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
	{
		w3[n4][n2][cy][cx]+=lambda*e3[n4]*cno2[n2][cy][cx];
	}

	for(n2=0;n2<NN_LAYER2;n2++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
	{
		z = e2[n2][cy][cx];
		lz = lambda*z;
		for(n1=0;n1<NN_LAYER1;n1++) for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
		{
			cw2[n2][n1][ry][rx]+=lz*cno1[n1][2*cy+ry][2*cx+rx];
		}
		thr2[n2]+=lz;
	}


	for(n1=0;n1<NN_LAYER1;n1++) for(cy=0;cy<L1W;cy++) for(cx=0;cx<L1W;cx++)
	{
		z = e1[n1][cy][cx];
		lz = lambda*z;
		for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
		{
			cw1[n1][ry][rx]+=lz*inpf[2*cy+ry][2*cx+rx];
		}
		thr1[n1]+=lz;
	}


	for(n4=0;n4<NN_OUTPUT;n4++) thro[n4]+=lambda*e3[n4];
}



//----------------------------------------------------------------------------------------
int cnn_LearnTo(int ndxtarg)
{
	int i, ndxo, iret = 0;
	double f1 = 1.0;
	double fth = 0.3;
	double z, vz, ferrm;

	nflerr = 0;
	for (i = 0; i < NN_OUTPUT; i++) t3[i] = 0;
	t3[ndxtarg] = 1.0;

	ndxo = cnn_Recogn();

	for (i = 0; i < NN_OUTPUT; i++)
	{
		e3[i] = t3[i] - l3[i];	
	}

	ferrm = t3[ndxtarg] - l3[ndxtarg];

	if (ndxo != ndxtarg)
	{
		nflerr = 1;
	}

	
//	if (ndxo != ndxtarg)	
	if (ndxo != ndxtarg || ferrm > fthre)	
	{
		iret = 1;
		back();
	}

	return iret;
}

void cnn_SerialiseParams()
{
    FILE *fo;

    fo=fopen("mnist_cnn_params.dat","wb");
    if (fo == NULL)
    {
        printf("Error! opening file");
        exit(1);
    }

    if(fwrite(thr1, sizeof(thr1[0]), NN_LAYER1, fo) != NN_LAYER1)
    {
        printf("Error! Writing thr1 to file");
        exit(1);
    }

    if(fwrite(thr2, sizeof(thr2[0]), NN_LAYER2, fo) != NN_LAYER2)
    {
        printf("Error! Writing thr2 to file");
        exit(1);
    }

    if(fwrite(thro, sizeof(thro[0]), NN_OUTPUT, fo) != NN_OUTPUT)
    {
        printf("Error! Writing thro to file");
        exit(1);
    }

    if(fwrite(cw1, sizeof(cw1[0][0][0]), NN_LAYER1*RFW*RFW, fo) != NN_LAYER1*RFW*RFW)
    {
        printf("Error! Writing cw1 to file");
        exit(1);
    }

    if(fwrite(cw2, sizeof(cw2[0][0][0][0]), NN_LAYER2*NN_LAYER1*RFW*RFW, fo) != NN_LAYER2*NN_LAYER1*RFW*RFW)
    {
        printf("Error! Writing cw2 to file");
        exit(1);
    }

    if(fwrite(w3, sizeof(w3[0][0][0][0]), NN_OUTPUT*NN_LAYER2*L2W*L2W, fo) != NN_OUTPUT*NN_LAYER2*L2W*L2W)
    {
        printf("Error! Writing w3 to file");
        exit(1);
    }

    fclose(fo);
}

void cnn_DeserialiseParams()
{
}