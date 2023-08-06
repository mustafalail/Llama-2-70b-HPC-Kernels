! OpenMP

! Prompt: "Create 3 code suggestions using the following parameters: Fortran OpenMP general matrix-matrix multiply subroutine."

! SUGGESTION 1 ! ----------------------------------

subroutine matrix_multiply(A, B, C, m, n, p) result(D)
    integer :: i, j, k
    real(kind=8), intent(in) :: A(m, n), B(n, p)
    real(kind=8), intent(out) :: D(m, p)
    
    !$OMP PARALLEL DO default(none) shared(A, B, D)
    do concurrent (i = 1:m, j = 1:p)
        do k = 1:n
            D(i, j) = D(i, j) + A(i, k) * B(k, j)
        enddo
    enddo
    !$OMP END PARALLEL DO
end subroutine matrix_multiply


! SUGGESTION 2 ! ----------------------------------

subroutine matrix_multiply(A, B, C, m, n, p) result(D)
    integer :: i, j
    real(kind=8), intent(in) :: A(m, n), B(n, p)
    real(kind=8), intent(out) :: D(m, p)
    
    !$OMP PARALLEL DO default(none) shared(A, B, D)
    do concurrent (i = 1:m)
        D(i, :) = zero
        do j = 1:n
            D(i, j) = D(i, j) + A(i, j) * B(j, :)
            !$OMP REDUCTION (+:D(i, j))
        enddo
    enddo
    !$OMP END PARALLEL DO
end subroutine matrix_multiply


! SUGGESTION 3 ! ----------------------------------

subroutine matrix_multiply(A, B, C, m, n, p) result(D)
    integer :: i, j, k
    real(kind=8), intent(in) :: A(m, n), B(n, p)
    real(kind=8), intent(out) :: D(m, p)
    
    if (m <= 1 .or. n <= 1 .or. p <= 1) return
    
    ! Divide matrices into smaller blocks
    integer :: block_size = min(m, n, p)
    real(kind=8), allocatable :: A_block(:, :), B_block(:, :), D_block(:, :)
    allocate(A_block(block_size, block_size), B_block(block_size, block_size), D_block(block_size, block_size))
    
    ! Initialize blocks
    do i = 1, m, block_size
        do j = 1, n, block_size
            do k = 1, p, block_size
                A_block(i, j) = A(i, j + (k - 1) * block_size)
                B_block(j, k) = B(j + (k - 1) * block_size, k)
                D_block(i, k) = zero
            enddo
            call matrix_multiply(A_block, B_block, D_block, block_size, block_size, block_size)
            do k = 1, p, block_size
                D(i, k) = D_block(i, k) + D(i, k + (block_size - 1) * block_size)
            enddo
        enddo
    enddo
    
    ! Free memory
    deallocate(A_block, B_block, D_block)
end subroutine matrix_multiply



! ---------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenMP(offload)

! Prompt: "Create 3 code suggestions using the following parameters: Fortran OpenMP(offload) general matrix-matrix multiply subroutine."

! SUGGESTION 1 ! ----------------------------------

SUBROUTINE matrix_multiply(A, B, C, m, n, p) RESULT(D)
   INTEGER :: i, j, k
   REAL(KIND=8), INTENT(IN) :: A(m, n), B(n, p)
   REAL(KIND=8), INTENT(OUT) :: D(m, p)
   REAL(KIND=8), ALLOCATABLE :: A_local(:, :), B_local(:, :)
   INTEGER :: num_threads, thread_id

   num_threads = OMP_GET_MAX_THREADS()
   thread_id = OMP_GET_THREAD_ID()

   ! Offload data to local arrays
   allocate(A_local(m, n))
   allocate(B_local(n, p))
   do i = 1, m
      do j = 1, n
         do k = 1, p
            A_local(i, j) = A(i, j + (k - 1) * n)
            B_local(j, k) = B(j + (k - 1) * n, k)
         enddo
      enddo
   enddo

   ! Perform matrix multiplication
   call omp_set_num_threads(num_threads)
   call omp_parallel_sections(omp_default_proc_bind())
   do i = 1, m
      do j = 1, p
         D(i, j) = zero
      enddo
   enddo
   do i = 1, m
      do j = 1, n
         do k = 1, p
            D(i, j) = D(i, j) + A_local(i, j) * B_local(j, k)
         enddo
      enddo
   enddo

   ! Synchronize threads
   call omp_barrier()

   ! Copy results back to main array
   do i = 1, m
      do j = 1, p
         D(i, j) = D(i, j) + D(i, j + (thread_id - 1) * m)
      enddo
   enddo

   ! Deallocate local arrays
   deallocate(A_local, B_local)
END SUBROUTINE matrix_multiply


! SUGGESTION 2 ! ----------------------------------

SUBROUTINE matrix_multiply(A, B, C, m, n, p) RESULT(D)
   INTEGER :: i, j, k
   REAL(KIND=8), INTENT(IN) :: A(m, n), B(n, p)
   REAL(KIND=8), INTENT(OUT) :: D(m, p)
   REAL(KIND=8), ALLOCATABLE :: A_local(:, :), B_local(:, :)
   INTEGER :: num_threads, thread_id
   INTEGER :: block_size = 1024

   num_threads = OMP_GET_MAX_THREADS()
   thread_id = OMP_GET_THREAD_ID()

   ! Offload data to local arrays in blocks
   allocate(A_local(m, n))
   allocate(B_local(n, p))
   do i = 1, m, block_size
      do j = 1, n, block_size
         do k = 1, p, block_size
            A_local(i, j) = A(i, j + (k - 1) * n)
            B_local(j, k) = B(j + (k - 1) * n, k)
         enddo
      enddo
   enddo

   ! Perform matrix multiplication in blocks
   call omp_set_num_threads(num_threads)
   call omp_parallel_sections(omp_default_proc_bind())
   do i = 1, m, block_size
      do j = 1, n, block_size
         do k = 1, p, block_size
            D(i, j) = D(i, j) + A_local(i, j) * B_local(j, k)
         enddo
      enddo
   enddo

   ! Synchronize threads
   call omp_barrier()

   ! Copy local results back to main array
   do i = 1, m
      do j = 1, p
         D(i, j) = D(i, j) + D(i, j + (thread_id - 1) * m)
      enddo
   enddo

   ! Deallocate local arrays
   deallocate(A_local, B_local)
END SUBROUTINE matrix_multiply


! SUGGESTION 3 ! ----------------------------------

SUBROUTINE matrix_multiply_strided(A, B, C, m, n, p, lda, ldb, ldc) RESULT(D)
   INTEGER :: i, j, k
   REAL(KIND=8), INTENT(IN) :: A(lda, n), B(ldb, p)
   REAL(KIND=8), INTENT(OUT) :: C(ldc, p)
   REAL(KIND=8), ALLOCATABLE :: D(:, :)

   allocate(D(ldc, p))

   !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i, j, k)
   do i = 1, m
      do j = 1, n
         do k = 1, p
            D(i, k) = D(i, k) + A(i, j) * B(j, k)
         enddo
      enddo
   enddo
   !$OMP END PARALLEL DO

END SUBROUTINE matrix_multiply_strided



! ---------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenACC

! Prompt: "Create 3 code suggestions using the following parameters: Fortran OpenACC general matrix-matrix multiply subroutine."

! SUGGESTION 1 ! ----------------------------------

SUBROUTINE matrix_multiply(A, B, C, m, n, p) RESULT(D)
   INTEGER :: i, j, k
   REAL(KIND=8), INTENT(IN) :: A(m, n), B(n, p)
   REAL(KIND=8), INTENT(OUT) :: C(m, p)
   REAL(KIND=8), ALLOCATABLE :: D(:, :)

   allocate(D(m, p))

   !$acc data copyin(A, B)
   !$acc data copyout(D)

   !$acc parallel num_gangs(4) dim(0:m-1, 0:n-1, 0:p-1)
   do i = 1, m
      do j = 1, n
         do k = 1, p
            D(i, k) = D(i, k) + A(i, j) * B(j, k)
         enddo
      enddo
   enddo

   !$acc end parallel

END SUBROUTINE matrix_multiply


! SUGGESTION 2 ! ----------------------------------

SUBROUTINE matrix_multiply_strided(A, B, C, m, n, p, lda, ldb, ldc) RESULT(D)
   INTEGER :: i, j, k
   REAL(KIND=8), INTENT(IN) :: A(lda, n), B(ldb, p)
   REAL(KIND=8), INTENT(OUT) :: C(ldc, p)
   REAL(KIND=8), ALLOCATABLE :: D(:, :)

   allocate(D(ldc, p))

   !$acc data copyin(A, B)
   !$acc data copyout(D)

   !$acc parallel num_gangs(4) dim(0:m-1, 0:n-1, 0:p-1)
   do i = 1, m
      do j = 1, n
         do k = 1, p
            D(i, k) = D(i, k) + A(i, j) * B(j, k)
         enddo
      enddo
   enddo

   !$acc end parallel

END SUBROUTINE matrix_multiply_strided


! SUGGESTION 3 ! ----------------------------------

SUBROUTINE matrix_multiply_tiled(A, B, C, m, n, p, tile_size) RESULT(D)
   INTEGER :: i, j, k, tid
   REAL(KIND=8), INTENT(IN) :: A(m, n), B(n, p)
   REAL(KIND=8), INTENT(OUT) :: C(m, p)
   REAL(KIND=8), ALLOCATABLE :: D(:, :)

   allocate(D(m, p))

   ! Define the tile size
   integer :: tilesize = 64

   !$acc enter data
   !$acc data copyin(A, B)
   !$acc data copyout(D)

   !$acc parallel num_gangs(4) dim(0:m-1, 0:n-1, 0:p-1)
   do i = 1, m
      do j = 1, n
         do k = 1, p
            ! Compute the tile index
            tid = (i - 1) / tilesize + 1

            !$acc loop gang
            do tile = 1, tilesize
               ! Compute the local tile indices
               i_loc = (tid - 1) * tilesize + tile
               j_loc = (j - 1) * tilesize + tile

               ! Compute the global tile indices
               i_glb = i_loc + (k - 1) * m
               j_glb = j_loc + (k - 1) * n

               ! Perform the matrix multiplication for the current tile
               D(i_glb, j_glb) = D(i_glb, j_glb) + A(i_loc, j_loc) * B(j_loc, k)
            enddo
         enddo
      enddo
   enddo

   !$acc end parallel

END SUBROUTINE matrix_multiply_tiled



! ---------------------------------------------------------------------------------------------------------------------------------------------------------

