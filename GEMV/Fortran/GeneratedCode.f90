! OpenMP

! Give me code for the following:
! Fortran, OpenMP, General matrix-vector multiply, subroutine

Program MatrixVectorMultiply
  Use, Intrinsic :: ISO_C_BINDING, Only : C_INT
  Implicit None

  ! Declare variables
  Real(C_FLOAT), Allocatable :: A(:, :)
  Real(C_FLOAT), Allocatable :: B(:)
  Real(C_FLOAT), Allocatable :: C(:)
  Integer :: i, j, n
  Logical :: thread_created = .false.

  ! Set up input parameters
  n = size(A, dimension=1)
  allocate(A(n, n))
  allocate(B(n))
  allocate(C(n))

  ! Initialize matrices and vector
  do concurrent (i=1:n)
    A(i,:) = reshape((/ (i-1)*10 + 1, 2*i-1, 3*i+1 /), shape(A))
    B(i) = i**2
  enddo

  ! Define OpenMP parallel region
  !$omp parallel num_threads(4) default(none) shared(A, B, C)

  ! Perform matrix-vector multiplication
  do concurrent (i=1:n)
    do concurrent (j=1:n)
      C(i) = C(i) + A(i,j)*B(j)
    enddo
  enddo

  !$omp end parallel

  ! Free memory
  deallocate(A, B, C)
end program MatrixVectorMultiply

! ------------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenMP (offload)

! Give me code for the following:
! Fortran, OpenMP (offload), General matrix-vector multiply, subroutine

Program MatrixVectorMultiply
  Use, Intrinsic :: ISO_C_BINDING, Only : C_INT
  Implicit None

  ! Declare variables
  Real(C_FLOAT), Allocatable :: A(:, :)
  Real(C_FLOAT), Allocatable :: B(:)
  Real(C_FLOAT), Allocatable :: C(:)
  Integer :: i, j, n
  Logical :: thread_created = .false.

  ! Set up input parameters
  n = size(A, dimension=1)
  allocate(A(n, n))
  allocate(B(n))
  allocate(C(n))

  ! Initialize matrices and vector
  do concurrent (i=1:n)
    A(i,:) = reshape((/ (i-1)*10 + 1, 2*i-1, 3*i+1 /), shape(A))
    B(i) = i**2
  enddo

  ! Define OpenMP parallel region with offloading
  !$omp parallel num_threads(4) default(none) shared(A, B, C) firstprivate(n)

  ! Offload computation to accelerator
  !$omp offload (A, B, C) proc(my_matrix_vector_multiply)

  ! Synchronize threads
  !$omp barrier

  ! Free memory
  deallocate(A, B, C)

contains

  subroutine my_matrix_vector_multiply(A, B, C, n) result(D)
    Real(C_FLOAT), Intent(in) :: A(n, n), B(n)
    Real(C_FLOAT), Intent(out) :: C(n)
    integer :: n

    ! Compute matrix-vector product
    do concurrent (i=1:n)
      do concurrent (j=1:n)
        D(i) = D(i) + A(i,j)*B(j)
      enddo
    enddo

    return
  end subroutine my_matrix_vector_multiply
end program MatrixVectorMultiply

! ------------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenACC

! Give me code for the following:
! Fortran, OpenACC, General matrix-vector multiply, subroutine

Program MatrixVectorMultiply
  Use, Intrinsic :: ISO_C_BINDING, Only : C_INT
  Implicit None

  ! Declare variables
  Real(C_FLOAT), Allocatable :: A(:, :)
  Real(C_FLOAT), Allocatable :: B(:)
  Real(C_FLOAT), Allocatable :: C(:)
  Integer :: i, j, n
  Logical :: thread_created = .false.

  ! Set up input parameters
  n = size(A, dimension=1)
  allocate(A(n, n))
  allocate(B(n))
  allocate(C(n))

  ! Initialize matrices and vector
  do concurrent (i=1:n)
    A(i,:) = reshape((/ (i-1)*10 + 1, 2*i-1, 3*i+1 /), shape(A))
    B(i) = i**2
  enddo

  ! Define OpenACC parallel region
  !$acc parallel num_gangs(4) default(none) shared(A, B, C) firstprivate(n)

  ! Offload computation to accelerator
  !$acc loop gang worker(A, B, C, n)

  ! Synchronize threads
  !$acc synchronize

  ! Free memory
  deallocate(A, B, C)

contains

  subroutine my_matrix_vector_multiply(A, B, C, n) result(D)
    Real(C_FLOAT), Intent(in) :: A(n, n), B(n)
    Real(C_FLOAT), Intent(out) :: C(n)
    integer :: n

    ! Compute matrix-vector product
    do concurrent (i=1:n)
      do concurrent (j=1:n)
        D(i) = D(i) + A(i,j)*B(j)
      enddo
    enddo

    return
  end subroutine my_matrix_vector_multiply
end program MatrixVectorMultiply
