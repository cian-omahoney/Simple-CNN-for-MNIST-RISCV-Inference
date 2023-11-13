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

float inpf[INPW][INPW];
float cno1[NN_LAYER1][L1W][L1W];
float cno2[NN_LAYER2][L2W][L2W];
unsigned char chrf[INPV_LEN];
float dgr[TBL_LEN];

float l3[NN_OUTPUT];

float cw1[NN_LAYER1][RFW][RFW];
float cw2[NN_LAYER2][NN_LAYER1][RFW][RFW];
float w3[NN_OUTPUT][NN_LAYER2][L2W][L2W];

float thr1[NN_LAYER1];
float thr2[NN_LAYER2];
float thro[NN_OUTPUT];

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

//--------------------------------------------------
void cnn_init(void)
{
	int i;
	float fv;
	float kz = 0.00390625;

	for(i=0;i<TBL_LEN;i++)
	{
		fv = (float) i;
		dgr[i] = kz*fv;
	}
}

//-------------------------------------------------
void cnn_DeserializeParams()
{
    FILE *fi;

    fi=fopen("mnist_cnn_params_float.dat","rb");
    if (fi == NULL)
    {
        printf("Error! opening file");
        exit(1);
    }

    if(fread(thr1, sizeof(thr1[0]), NN_LAYER1, fi) != NN_LAYER1)
    {
        printf("Error! Reading thr1 to file");
        exit(1);
    }

    if(fread(thr2, sizeof(thr2[0]), NN_LAYER2, fi) != NN_LAYER2)
    {
        printf("Error! Reading thr2 to file");
        exit(1);
    }

    if(fread(thro, sizeof(thro[0]), NN_OUTPUT, fi) != NN_OUTPUT)
    {
        printf("Error! Reading thro to file");
        exit(1);
    }

    if(fread(cw1, sizeof(cw1[0][0][0]), NN_LAYER1*RFW*RFW, fi) != NN_LAYER1*RFW*RFW)
    {
        printf("Error! Reading cw1 to file");
        exit(1);
    }

    if(fread(cw2, sizeof(cw2[0][0][0][0]), NN_LAYER2*NN_LAYER1*RFW*RFW, fi) != NN_LAYER2*NN_LAYER1*RFW*RFW)
    {
        printf("Error! Reading cw2 to file");
        exit(1);
    }

    if(fread(w3, sizeof(w3[0][0][0][0]), NN_OUTPUT*NN_LAYER2*L2W*L2W, fi) != NN_OUTPUT*NN_LAYER2*L2W*L2W)
    {
        printf("Error! Reading w3 to file");
        exit(1);
    }

    fclose(fi);
}


//-------------------------------------------------
int cnn_Recogn(void)
{
	int iret=0;
	float fsum,z;
	int i,j,n,nn,cx,cy,rx,ry;
	float sq[NN_OUTPUT];
	float se[NN_OUTPUT];
	float qmax = 10000000.0;
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