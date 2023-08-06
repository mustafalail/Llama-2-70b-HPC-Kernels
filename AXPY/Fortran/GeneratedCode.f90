! OpenMP

! Prompt: "Create 3 code suggestions using the following parameters: Fortran OpenMP AXPY subroutine."

! SUGGESTION 1 ! ----------------------------------

SUBROUTINE axpy_omp(a, x, y) result(y)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: y
    integer :: i
    !$OMP PARALLEL DO default(none) shared(x, y, a)
    do concurrent (i = 1:size(x)) y(i) = y(i) + a * x(i)
    !$OMP END PARALLEL DO
END SUBROUTINE axpy_omp


! SUGGESTION 2 ! ----------------------------------

SUBROUTINE axpy_omp_dma(a, x, y) result(y)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(in) :: x
    real(kind=8), allocatable :: y(:)
    integer :: i
    !$OMP PARALLEL DO default(none) shared(x, y, a)
    allocate(y(size(x)))
    do concurrent (i = 1:size(x)) y(i) = y(i) + a * x(i)
    !$OMP END PARALLEL DO
END SUBROUTINE axpy_omp_dma


! SUGGESTION 3 ! ----------------------------------

SUBROUTINE axpy_omp_tp(a, x, y) result(y)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(in) :: x
    real(kind=8), allocatable :: y(:) thread private
    integer :: i
    !$OMP PARALLEL DO default(none) shared(x, a) firstprivate(y)
    do concurrent (i = 1:size(x)) y(i) = y(i) + a * x(i)
    !$OMP END PARALLEL DO
END SUBROUTINE axpy_omp_tp



! ---------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenMP(offload)

! Prompt: "Create 3 code suggestions using the following parameters: Fortran OpenMP(offload) AXPY subroutine."

! SUGGESTION 1 ! ----------------------------------

SUBROUTINE axpy_offload(a, x, y) result(y)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: y
    integer :: i
    !$OMP OFFLOAD(y)
    !$OMP PARALLEL DO default(none) shared(x, a)
    do concurrent (i = 1:size(x)) y(i) = y(i) + a * x(i)
    !$OMP END PARALLEL DO
END SUBROUTINE axpy_offload


! SUGGESTION 2 ! ----------------------------------

SUBROUTINE axpy_strided_offload(a, x, y) result(y)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: y
    integer :: i
    integer :: stride = size(x)/4
    !$OMP OFFLOAD(y)
    !$OMP PARALLEL DO default(none) shared(x, a)
    do concurrent (i = 1:size(x):stride) y(i) = y(i) + a * x(i)
    !$OMP END PARALLEL DO
END SUBROUTINE axpy_strided_offload


! SUGGESTION 3 ! ----------------------------------

SUBROUTINE axpy_blocked_offload(a, x, y) result(y)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: y
    integer :: i
    integer :: blocksiz = 64
    integer :: numblks = ceiling(size(x)/blocksiz)
    !$OMP OFFLOAD(y)
    !$OMP PARALLEL DO default(none) shared(x, a)
    do concurrent (i = 1:numblks) &
         y(i*blocksiz:(i+1)*blocksiz-1) = &
         y(i*blocksiz:(i+1)*blocksiz-1) + a * x(i*blocksiz:(i+1)*blocksiz-1)
    !$OMP END PARALLEL DO
END SUBROUTINE axpy_blocked_offload



! ---------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenACC

! Prompt: "Create 3 code suggestions using the following parameters: Fortran OpenACC AXPY subroutine."

! SUGGESTION 1 ! ----------------------------------

SUBROUTINE axpy_openacc(a, x, y) result(y)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: y
    integer :: i
    !DECLARE_GROUP(gang)
    !SET_GROUP(gang, omp_get_max_threads())
    !$omp parallel num_threads(gang) default(none) shared(x, a)
    do concurrent (i = 1:size(x)) y(i) = y(i) + a * x(i)
    !$omp end parallel
END SUBROUTINE axpy_openacc


! SUGGESTION 2 ! ----------------------------------

SUBROUTINE axpy_strided_openacc(a, x, y) result(y)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: y
    integer :: i
    integer :: stride = size(x)/4
    !DECLARE_GROUP(gang)
    !SET_GROUP(gang, omp_get_max_threads())
    !$omp parallel num_threads(gang) default(none) shared(x, a)
    do concurrent (i = 1:size(x):stride) y(i) = y(i) + a * x(i)
    !$omp end parallel
END SUBROUTINE axpy_strided_openacc


! SUGGESTION 3 ! ----------------------------------

SUBROUTINE axpy_blocked_openacc(a, x, y) result(y)
    real(kind=8), intent(in) :: a
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: y
    integer :: i
    integer :: blocksiz = 64
    integer :: numblks = ceiling(size(x)/blocksiz)
    !DECLARE_GROUP(gang)
    !SET_GROUP(gang, omp_get_max_threads())
    !$omp parallel num_threads(gang) default(none) shared(x, a)
    do concurrent (i = 1:numblks) &
         y(i*blocksiz:(i+1)*blocksiz-1) = &
         y(i*blocksiz:(i+1)*blocksiz-1) + a * x(i*blocksiz:(i+1)*blocksiz-1)
    !$omp end parallel
END SUBROUTINE axpy_blocked_openacc



! ---------------------------------------------------------------------------------------------------------------------------------------------------------

