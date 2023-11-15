#include <math.h>
#include <stdio.h>
#include <stdint-gcc.h>
#include "mnist_params.h"
#include "mnist_params_fixedPoint.h"

#define RFW			6
#define INPW		28
#define L1W			12
#define L2W			4
#define NN_LAYER1	12
#define NN_LAYER2	32
#define NN_OUTPUT	10
#define INPV_LEN	784
#define TBL_LEN		256

float inpf[INPW][INPW];
int32_t cno1[NN_LAYER1][L1W][L1W];
int32_t cno2[NN_LAYER2][L2W][L2W];
uint8_t chrf[INPV_LEN];
float dgr[TBL_LEN];
int32_t l3[NN_OUTPUT];

//--------------------------------------------------
void cnn_init(void)
{
	uint16_t i;
	float kz = 0.00390625;

	for(i=0;i<TBL_LEN;i++)
	{
		dgr[i] = kz*i;
	}
}

//-------------------------------------------------
int cnn_Recogn(void)
{
	int32_t iret=0;
	int32_t fsum,z;
    int32_t sq[NN_OUTPUT];
    int32_t se[NN_OUTPUT];
    uint32_t i,j,n,nn,cx,cy,rx,ry;
	int32_t qmax = 100000000;
	uint8_t chx;

	for(i=0;i<INPW;i++) for(j=0;j<INPW;j++)
	{
		chx = chrf[i*INPW+j];
		inpf[i][j] = dgr[chx];
	}

//    for(int ry=0;ry<28;ry++)
//    {
//        for(int rx=0;rx<28;rx++)
//        {
//            if(inpf[ry][rx] > 0) printf("[x]");
//            else                  printf("  ");
//        }
//        printf("\n");
//    }


    for(n=0;n<NN_LAYER1;n++) for(cy=0;cy<L1W;cy++) for(cx=0;cx<L1W;cx++)
	{
		fsum=thr1_fp[n];
		for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
		{
			fsum+=cw1_fp[n][ry][rx]*inpf[2*cy+ry][2*cx+rx];
		}
		if(fsum>0) z = fsum;
		else	   z = 0;	
		cno1[n][cy][cx] = z;
	}


	for(n=0;n<NN_LAYER2;n++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
	{
		fsum=thr2_fp[n];
		for(nn=0;nn<NN_LAYER1;nn++) for(ry=0;ry<RFW;ry++) for(rx=0;rx<RFW;rx++)
		{
			fsum+=cw2_fp[n][nn][ry][rx]*cno1[nn][2*cy+ry][2*cx+rx];
		}
		if(fsum>0) z = fsum;
		else	   z = 0;	
		cno2[n][cy][cx] = z;
	}


	for(n=0;n<NN_OUTPUT;n++)
	{
		fsum=thro_fp[n];
		for(nn=0;nn<NN_LAYER2;nn++) for(cy=0;cy<L2W;cy++) for(cx=0;cx<L2W;cx++)
		{
			fsum+=w3_fp[n][nn][cy][cx]*cno2[nn][cy][cx];
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