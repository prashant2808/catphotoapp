 #include<stdio.h>
#include<conio.h>
void main()
{
clrscr();
int i,t,a[100],d,n;
printf("Enter The No. Of Elements\n");
scanf("%d",&n);
printf("Enter %d Integers\n",n);
for(i=0;i<n;i++)
{
  scanf("%d",&a[i]);
}
for(i=1;i<=n-1;i++)
{
d=i;
while(d>0&&a[d]<a[d-1])
{ t=a[d];
a[d]=a[d-1];
a[d-1]=t;
d--;
}
}
printf("Sorted List in Ascending Order is:\n");
for(i=0;i<=n-1;i++)
{
printf("%d\t",a[i]);
}
getch();
}