#ifndef A8FF70C1_567E_449A_8DBD_3F0FC21EC10B
#define A8FF70C1_567E_449A_8DBD_3F0FC21EC10B


    template <size_t H, size_t W, bool INIT>
    inline void MultMatMatKernel(size_t Aw, double *Ai, size_t Adist, double *Bj, size_t Bdist, double *Cij, size_t Cdist) noexcept
    {
        if constexpr (!H || !W)
            return;
        else
        {
            ASC_HPC::SIMD<double, W> sum[H];
            for (size_t l = 0; l < H; ++l)
            {
                if constexpr (!INIT)
                    sum[l] = ASC_HPC::SIMD<double, W>(Cij + l * Cdist);
                else
                    sum[l] = ASC_HPC::SIMD<double, W>(0.0);
            }
            for (size_t k = 0; k < Aw; ++k)
            {
                ASC_HPC::SIMD<double, W> y1(Bj + k * Bdist);

                for (size_t l = 0; l < H; ++l)
                    sum[l] = ASC_HPC::FMA(ASC_HPC::SIMD<double, W>(*(Ai + k * Adist + l)), y1, sum[l]);
            }
            for (size_t l = 0; l < H; ++l)
            {
                sum[l].Store(Cij + l * Cdist);
            }
        }
    }

    template <size_t H, size_t W, bool INIT>
    void MultMatMat2Timed(MatrixView<double, ColMajor> A[], MatrixView<double, RowMajor> largeA, size_t i1, size_t i2, size_t j1, size_t j2, MatrixView<double, RowMajor> largeB, MatrixView<double, RowMajor> C)
    {
        {
        //block for lifetime of B, firstW
        size_t firstW = std::min(C.nCols(), W);
        alignas(64) double memB[W * ABLOCK_HEIGHT];
        MatrixView<double, RowMajor> B(largeB.nRows(), firstW, firstW, memB);
        static ASC_HPC::Timer tb("pack B micropanel", { 1, 0, 0});
        tb.Start();
        B = largeB.Cols(0,firstW); //j2<W?
        tb.Stop();
        //copy Ablock using all threads
        ASC_HPC::TaskManager::RunParallel([&](int id, int numThreads)
        {
            size_t j =0;
            size_t i = id*H;
            for (; i + H <= C.nRows(); i += H*numThreads){
                {
                    static ASC_HPC::Timer ta("pack A micropanel", { 1, 1, 0});
                    ta.Start();
                    A[i/H].Cols(0,j2-j1) = largeA.Rows(i1+i, i1+i+H).Cols(j1, j2);
                    ta.Stop();
                }
                static ASC_HPC::Timer tk("Microkernel "+std::to_string(H)+"x"+std::to_string(firstW), { 0, 1, 0});
                tk.Start();
                (*dispatch_MatMatMult[H][firstW][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
                tk.Stop();
            }
            if(i<C.nRows()&&i+H>C.nRows()){
                static ASC_HPC::Timer ta("pack A micropanel", { 1, 1, 0});
                ta.Start();
                A[i/H].Rows(0,C.nRows()-i).Cols(0,j2-j1) = largeA.Rows(i1+i, i2).Cols(j1, j2);
                ta.Stop();

                static ASC_HPC::Timer tk("MicrokernelDi "+std::to_string(C.nRows()-i)+"x"+std::to_string(firstW), { 0, 1, 0});
                tk.Start();
                (*dispatch_MatMatMult[C.nRows()-i][firstW][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
                tk.Stop();
            }
        });
        }

        //sync threads before starting to work with cached Ablock
        ASC_HPC::TaskManager::RunParallel([&](int id, int numThreads)
        { 
        size_t j = id*W+W;
        size_t i;

        alignas(64) double memB[W * ABLOCK_HEIGHT];

        for (; j + W <= C.nCols(); j += W*numThreads){
            static ASC_HPC::Timer tb("pack B micropanel", { 1, 0, 0});
            tb.Start();
            MatrixView<double, RowMajor> B(largeB.nRows(), W, W, memB);
            B = largeB.Cols(j,j+W);
            tb.Stop();
            for (i=0; i + H < C.nRows(); i += H){
                static ASC_HPC::Timer tk("Microkernel "+std::to_string(H)+"x"+std::to_string(W), { 0, 1, 0});
                tk.Start();
                MultMatMatKernel<H, W, INIT>(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
                tk.Stop();
            }

            static ASC_HPC::Timer tk("MicrokernelDi "+std::to_string(C.nRows()-i)+"x"+std::to_string(W), { 0, 1, 0});
            tk.Start();
            (*dispatch_MatMatMult[C.nRows()-i][W][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
            tk.Stop();
        }
        if(j<C.nCols()&&j+W>C.nCols()){
            static ASC_HPC::Timer tb("pack B micropanel", { 1, 0, 0});
            tb.Start();
            MatrixView<double, RowMajor> B(largeB.nRows(), C.nCols()-j, C.nCols()-j, memB);
            B = largeB.Cols(j,C.nCols());
            tb.Stop();
            for (i=0; i + H <= C.nRows(); i += H){
                static ASC_HPC::Timer tk("MicrokernelDi "+std::to_string(H)+"x"+std::to_string(C.nCols()-j), { 0, 1, 0});
                tk.Start();
                //std::cout << A[i/H].Cols(0,j2-j1) << std::endl;
                (*dispatch_MatMatMult[H][C.nCols()-j][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
                tk.Stop();
            }
            static ASC_HPC::Timer tk("MicrokernelDi "+std::to_string(C.nRows()-i)+"x"+std::to_string(C.nCols()-j), { 0, 1, 0});
            tk.Start();
            (*dispatch_MatMatMult[C.nRows()-i][C.nCols()-j][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
            tk.Stop();
        } }); 
    }

    template <size_t H, size_t W, bool INIT>
    void MultMatMat2(MatrixView<double, ColMajor> A[], MatrixView<double, RowMajor> largeA, size_t i1, size_t i2, size_t j1, size_t j2, MatrixView<double, RowMajor> largeB, MatrixView<double, RowMajor> C)
    {
        {
        //block for lifetime of B, firstW
        size_t firstW = std::min(C.nCols(), W);
        alignas(64) double memB[W * ABLOCK_HEIGHT];
        MatrixView<double, RowMajor> B(largeB.nRows(), firstW, firstW, memB);
        B = largeB.Cols(0,firstW); //j2<W?
        //copy Ablock using all threads
        ASC_HPC::TaskManager::RunParallel([&](int id, int numThreads)
        {
            size_t j =0;
            size_t i = id*H;
            for (; i + H <= C.nRows(); i += H*numThreads){
                A[i/H].Cols(0,j2-j1) = largeA.Rows(i1+i, i1+i+H).Cols(j1, j2);
            
                (*dispatch_MatMatMult[H][firstW][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
            }
            if(i<C.nRows()&&i+H>C.nRows()){
                A[i/H].Rows(0,C.nRows()-i).Cols(0,j2-j1) = largeA.Rows(i1+i, i2).Cols(j1, j2);
                (*dispatch_MatMatMult[C.nRows()-i][firstW][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
            }
        });
        }
        
        //sync threads before starting to work with cached Ablock
        ASC_HPC::TaskManager::RunParallel([&](int id, int numThreads)
        { 
        size_t j = id*W+W;
        size_t i;

        alignas(64) double memB[W * ABLOCK_HEIGHT];

        for (; j + W <= C.nCols(); j += W*numThreads){
            MatrixView<double, RowMajor> B(largeB.nRows(), W, W, memB);
            B = largeB.Cols(j,j+W);
            for (i=0; i + H < C.nRows(); i += H)
                MultMatMatKernel<H, W, INIT>(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());

            (*dispatch_MatMatMult[C.nRows()-i][W][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
        }
        if(j<C.nCols()&&j+W>C.nCols()){
            MatrixView<double, RowMajor> B(largeB.nRows(), C.nCols()-j, C.nCols()-j, memB);
            B = largeB.Cols(j,C.nCols());
            for (i=0; i + H <= C.nRows(); i += H)
                (*dispatch_MatMatMult[H][C.nCols()-j][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
            
            (*dispatch_MatMatMult[C.nRows()-i][C.nCols()-j][INIT])(j2-j1, &A[i/H](0,0), A[i/H].Dist(), &B(0, 0), B.Dist(), &C(i, j), C.Dist());
        } });      
    }

    void MultMatMat(MatrixView<double, RowMajor> A, MatrixView<double, RowMajor> B, MatrixView<double, RowMajor> C)
    {
        size_t i1 = 0;
        alignas(64) double memBA[ABLOCK_HEIGHT / KERNEL_HEIGHT][KERNEL_HEIGHT * ABLOCK_WIDTH];
        MatrixView<double, ColMajor> Ablock[ABLOCK_HEIGHT / KERNEL_HEIGHT];
        for (size_t i = 0; i < ABLOCK_HEIGHT / KERNEL_HEIGHT; i++)
            Ablock[i] = MatrixView<double, ColMajor>(KERNEL_HEIGHT, ABLOCK_WIDTH, KERNEL_HEIGHT, memBA[i]);

        for (; i1 < A.nRows(); i1 += ABLOCK_HEIGHT)
        {
            size_t j1 = 0;
            for (; j1 < A.nCols(); j1 += ABLOCK_WIDTH)
            {
                size_t i2 = std::min(A.nRows(), i1 + ABLOCK_HEIGHT);
                size_t j2 = std::min(A.nCols(), j1 + ABLOCK_WIDTH);
                if (!j1)
                {
                    if (ASC_HPC::TaskManager::writeTrace)
                        MultMatMat2Timed<KERNEL_HEIGHT, KERNEL_WIDTH, true>(Ablock, A, i1, i2, j1, j2, B.Rows(j1, j2), C.Rows(i1, i2));
                    else
                        MultMatMat2<KERNEL_HEIGHT, KERNEL_WIDTH, true>(Ablock, A, i1, i2, j1, j2, B.Rows(j1, j2), C.Rows(i1, i2));
                }   
                else
                {
                    if (ASC_HPC::TaskManager::writeTrace)
                        MultMatMat2Timed<KERNEL_HEIGHT, KERNEL_WIDTH, false>(Ablock, A, i1, i2, j1, j2, B.Rows(j1, j2), C.Rows(i1, i2));
                    else
                        MultMatMat2<KERNEL_HEIGHT, KERNEL_WIDTH, false>(Ablock, A, i1, i2, j1, j2, B.Rows(j1, j2), C.Rows(i1, i2));
                }
            }
        }
    }

    template <typename T, ORDERING ORD>
    Matrix<T, ORD> InnerProduct(MatrixView<T, ORD> &m1, MatrixView<T, ORD> &m2)
    {
        Matrix<T, RowMajor> res(m1.nRows(), m2.nCols());
        MultMatMat(m1, m2, res);
        return res;
    }

    template <size_t I, size_t J>
    void InnerIteratorDispatch()
    {
        dispatch_MatMatMult[I][J][0] = &MultMatMatKernel<I, J>;
        dispatch_MatMatMult[I][J][1] = &MultMatMatKernel<I, J, true>;

        if constexpr (J != 0)
            InnerIteratorDispatch<I, J - 1>();
    }

    template <size_t I, size_t J>
    void IteratorDispatch()
    {
        InnerIteratorDispatch<I, J>();

        if constexpr (I != 0)
            IteratorDispatch<I - 1, J>();
    }

    auto init_MatMatMult = []()
    {
        IteratorDispatch<KERNEL_HEIGHT, KERNEL_WIDTH>();

        return 1;
    }();

#endif /* A8FF70C1_567E_449A_8DBD_3F0FC21EC10B */
