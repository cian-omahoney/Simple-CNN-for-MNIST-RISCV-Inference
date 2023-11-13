#include <time.h>
#include <stdlib.h>
#include <stdio.h>

char mnistDirectory[] = "../mnist/";
char sz_Tstdata[] = "t10k-images-idx3-ubyte";
char sz_Tstlabels[] = "t10k-labels-idx1-ubyte";

int n_images;
int n_tst_images;
unsigned char * pb_tst_data;
int *pi_tst_labels;

extern unsigned char chrf[784];

void cnn_init(void);
int cnn_Recogn(void);
void cnn_print_network_info(FILE *fp);
void cnn_DeserializeParams(void);

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

int read_mnist(void)
{
    char bbuf[512];
    int n = 0;

    sprintf(bbuf,"%s%s",mnistDirectory,sz_Tstlabels);
    pi_tst_labels = read_mnist_labels(bbuf);

    n_tst_images = n_images;

    sprintf(bbuf,"%s%s",mnistDirectory,sz_Tstdata);
    pb_tst_data = read_mnist_data(bbuf);

    if(pi_tst_labels!=0 && pb_tst_data!=0)
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

void empty_arrays(void)
{
	if(pb_tst_data)
	{
		free(pb_tst_data);
	}
	if(pi_tst_labels)
	{
		free(pi_tst_labels);
	}
}

void do_experiment()
{
    time_t ltime;
    int numerrtst;
    int i,j,k,l;
    int nrec;
    int staterr[10];
    FILE *fo;

    unsigned char *pbtst;

	cnn_init();
    cnn_DeserializeParams();

	fo=fopen("mnist_demo_log.txt","a+");

	cnn_print_network_info(fo);

	for(j=0;j<10;j++) staterr[j]=0;

	printf("Recognition\n");
	fprintf(fo,"Recognition\n");
	numerrtst = 0;
	for(j=0;j<n_tst_images;j++) {
        pbtst = pb_tst_data + j * 784;
        for (l = 0; l < 784; l++) chrf[l] = *(pbtst + l);

        nrec = cnn_Recogn();
        if (nrec != pi_tst_labels[j]) {
            ++numerrtst;
            k = pi_tst_labels[j];
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
