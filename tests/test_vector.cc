#include <iostream>

#include <vector.h>
#include <complex>

int main()
{
    size_t n = 5;
    bla::Vector<double> x(n), y(n);
    bla::Vector<std::complex<double>> a(n);

    for (size_t i = 0; i < x.Size(); i++)
    {
        x(i) = i;
        y(i) = 10;
        a(i) = {(double)i, 1};
    }

    bla::Vector<double> z = x + y;

    bla::Vector<std::complex<double>> b = a + y;

    std::cout << "x+y = " << z << std::endl;
    std::cout << "a+y = " << b << std::endl;
    std::cout << "a = " << a << std::endl;
}
