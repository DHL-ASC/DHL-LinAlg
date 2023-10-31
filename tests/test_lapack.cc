#include <iostream>

#include <vector.h>
#include <lapack_interface.h>


using namespace ASC_bla;
using namespace std;


int main()
{
  Vector<double> x(5);
  Vector<double> y(5);

  for (int i = 0; i < x.Size(); i++)
    {
      x(i) = i;
      y(i) = 2;
    }

  cout << "x = " << x << endl;
  cout << "y = " << y << endl;
  
  AddVectorLapack (2, x, y);  
  cout << "y+2*x = " << y << endl;
}

  
