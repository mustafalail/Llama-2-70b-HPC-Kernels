! OpenMP

! Prompt: "Give me code for the following:
! Fortran, OpenMP, General matrix-matrix multiply, subroutine"

Program MatrixMultiply
  Use ISO_C_BINDING
  Implicit none

  ! Declare variables
  Real(C_DOUBLE), allocatable :: A(:, :)
  Real(C_DOUBLE), allocatable :: B(:, :)
  Real(C_DOUBLE), allocatable :: C(:, :)
  integer :: ierr
  integer :: n = size(A, dimension=1)
  integer :: m = size(A, dimension=2)
  integer :: p = size(B, dimension=1)
  integer :: q = size(B, dimension=2)
  integer :: r = size(C, dimension=1)
  integer :: s = size(C, dimension=2)

  ! Allocate memory for matrices
  allocate(A(n,m))
  allocate(B(p,q))
  allocate(C(r,s))

  ! Initialize matrices
  do concurrent (i=1:n, j=1:m) A(i,j) = 0.0d0
  do concurrent (i=1:p, j=1:q) B(i,j) = 0.0d0
  do concurrent (i=1:r, j=1:s) C(i,j) = 0.0d0

  ! Define subroutine for matrix-matrix multiplication
  subroutine mmul(A, B, C, n, m, p, q, r, s) result(D)
    implicit none
    real(c_double), intent(in) :: A(n, m)
    real(c_double), intent(in) :: B(p, q)
    real(c_double), intent(out) :: C(r, s)
    integer :: n, m, p, q, r, s
    !$omp parallel num_threads(4) default(none) shared(A, B, C)
      ! Perform matrix-matrix multiplication
      do concurrent (i=1:n, j=1:q, k=1:p) C(i,j) = C(i,j) + A(i,k)*B(k,j)
    !$omp end parallel
  contains
    ! Function to perform matrix-vector multiplication
    function dotProduct(A, B, n) result(dot)
      implicit none
      real(c_double), intent(in) :: A(n)
      real(c_double), intent(in) :: B(n)
      integer :: n
      dot = sum(A*B)
    end function dotProduct
  end subroutine mmul

  ! Call subroutine
  call mmul(A, B, C, n, m, p, q, r, s)

  ! Deallocate memory for matrices
  deallocate(A, B, C)
end program MatrixMultiply

! -----------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenMP (offload)

! Prompt: "Give me code for the following:
! Fortran, OpenMP (offload), General matrix-matrix multiply, subroutine"

Program MatrixMultiply
  Use ISO_C_BINDING
  Implicit none

  ! Declare variables
  Real(C_DOUBLE), allocatable :: A(:, :)
  Real(C_DOUBLE), allocatable :: B(:, :)
  Real(C_DOUBLE), allocatable :: C(:, :)
  integer :: ierr
  integer :: n = size(A, dimension=1)
  integer :: m = size(A, dimension=2)
  integer :: p = size(B, dimension=1)
  integer :: q = size(B, dimension=2)
  integer :: r = size(C, dimension=1)
  integer :: s = size(C, dimension=2)

  ! Allocate memory for matrices
  allocate(A(n,m))
  allocate(B(p,q))
  allocate(C(r,s))

  ! Initialize matrices
  do concurrent (i=1:n, j=1:m) A(i,j) = 0.0d0
  do concurrent (i=1:p, j=1:q) B(i,j) = 0.0d0
  do concurrent (i=1:r, j=1:s) C(i,j) = 0.0d0

  ! Define subroutine for matrix-matrix multiplication
  subroutine mmul(A, B, C, n, m, p, q, r, s) result(D)
    implicit none
    real(c_double), intent(in) :: A(n, m)
    real(c_double), intent(in) :: B(p, q)
    real(c_double), intent(out) :: C(r, s)
    integer :: n, m, p, q, r, s
    !$omp declare mapper(present)
    !$omp declare target(present)
    !$omp teams distribute parallel do concurrent(4) default(none) shared(A, B, C)
    !$omp end declare target
  contains
    ! Function to perform matrix-vector multiplication
    function dotProduct(A, B, n) result(dot)
      implicit none
      real(c_double), intent(in) :: A(n)
      real(c_double), intent(in) :: B(n)
      integer :: n
      dot = sum(A*B)
    end function dotProduct
  end subroutine mmul

  ! Offload computation to GPU
  call omp_set_num_teams(4)
  call omp_set_team_size(4)
  call mmul(A, B, C, n, m, p, q, r, s)

  ! Synchronize threads
  call omp_barrier()

  ! Deallocate memory for matrices
  deallocate(A, B, C)
end program MatrixMultiply

! -----------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenACC

! Prompt: "Give me code for the following:
! Fortran, OpenACC, General matrix-matrix multiply, subroutine"

program main
  use, intrinsic :: iso_c_binding
  use :: acc
  implicit none

  type(acc::array) :: A, B, C
  integer :: n, m, p, q, r, s
  integer :: err

  ! Declare variables
  n = size(A, dim=1)
  m = size(A, dim=2)
  p = size(B, dim=1)
  q = size(B, dim=2)
  r = size(C, dim=1)
  s = size(C, dim=2)

  ! Allocate memory for matrices
  allocate(A(n, m))
  allocate(B(p, q))
  allocate(C(r, s))

  ! Initialize matrices
  do concurrent (i=1:n, j=1:m) A(i,j) = 0.0d0
  do concurrent (i=1:p, j=1:q) B(i,j) = 0.0d0
  do concurrent (i=1:r, j=1:s) C(i,j) = 0.0d0

  ! Define subroutine for matrix-matrix multiplication
  subroutine mmul(A, B, C, n, m, p, q, r, s) result(D)
    implicit none
    real(c_double), intent(in) :: A(n, m)
    real(c_double), intent(in) :: B(p, q)
    real(c_double), intent(out) :: C(r, s)
    integer :: n, m, p, q, r, s
    !$omp declare mapper(present)
    !$omp declare target(present)
    !$omp teams distribute parallel do concurrent(4) default(none) shared(A, B, C)
    !$omp end declare target
  contains
    ! Function to perform matrix-vector multiplication
    function dotProduct(A, B, n) result(dot)
      implicit none
      real(c_double), intent(in) :: A(n)
      real(c_double), intent(in) :: B(n)
      integer :: n
      dot = sum(A*B)
    end function dotProduct

    ! Perform matrix-matrix multiplication
    do concurrent (i=1:r, j=1:s)
       C(i,j) = dotProduct(A(i,:), B(:,j), m)
    end do
  end subroutine mmul

  ! Call subroutine
  call mmul(A, B, C, n, m, p, q, r, s)

  ! Synchronize threads
  call omp_barrier()

  ! Deallocate memory for matrices
  deallocate(A, B, C)
end program main
