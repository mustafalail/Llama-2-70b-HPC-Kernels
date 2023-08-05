! OpenMP

! Prompt: "Give me code for the following:
! Fortran, OpenMP, General matrix-vector multiply, subroutine"

! VERSION 1 !


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


! VERSION 2 !


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
  !$omp parallel num_threads(4) default(none) shared(A, B, C) firstprivate(n)

  ! Offload computation to accelerator
  !$omp for schedule(static) nowait
  do i = 1, n
    do j = 1, n
      C(i) = C(i) + A(i,j)*B(j)
    enddo
  enddo

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



! VERSION 3 !

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
  !$omp parallel num_threads(4) default(none) shared(A, B, C) firstprivate(n)

  ! Offload computation to accelerator
  !$omp for schedule(static) nowait
  do i = 1, n
    do j = 1, n
      C(i) = C(i) + A(i,j)*B(j)
    enddo
  enddo

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

! Main program
call my_matrix_vector_multiply(A, B, C, n)

! Print results
do i = 1, n
  write (*,*) C(i)
enddo

end program MatrixVectorMultiply

! ------------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenMP (offload)

! Prompt: "Give me code for the following:
! Fortran, OpenMP (offload), General matrix-vector multiply, subroutine"


! VERSION 1 !

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


! VERSION 2 !

module my_modules
  implicit none

 contains

  subroutine my_matrix_vector_multiply(A, B, C, n) result(D)
    real(c_float), intent(in) :: A(n, n), B(n)
    real(c_float), intent(out) :: C(n)
    integer :: n

    ! Declare OpenMP variables
    integer :: tid, nthreads

    ! Set up OpenMP environment
    call omp_set_num_threads(4)
    call omp_set_dynamic(0)

    ! Offload computation to accelerator
    !$omp parallel num_threads(4) default(none) shared(A, B, C, n) firstprivate(tid)

    ! Compute matrix-vector product
    do concurrent (i=1:n)
      do concurrent (j=1:n)
        C(i) = C(i) + A(i,j)*B(j)
      enddo
    enddo

    ! Synchronize threads
    !$omp barrier

    ! Return
    return
  end subroutine my_matrix_vector_multiply

end module my_modules


! VERSION 3 !

module my_modules
  implicit none

 contains

  subroutine my_matrix_vector_multiply(A, B, C, n) result(D)
    real(c_float), intent(in) :: A(n, n), B(n)
    real(c_float), intent(out) :: C(n)
    integer :: n

    ! Declare OpenMP variables
    integer :: tid, nthreads

    ! Set up OpenMP environment
    call omp_init_parallel()
    call omp_set_max_threads(4)
    call omp_set_thread_limit(4)

    ! Offload computation to accelerator
    !$omp parallel num_threads(4) default(none) shared(A, B, C, n) firstprivate(tid)

    ! Compute matrix-vector product
    do concurrent (i=1:n)
      do concurrent (j=1:n)
        C(i) = C(i) + A(i,j)*B(j)
      enddo
    enddo

    ! Synchronize threads
    !$omp barrier

    ! Return
    return
  end subroutine my_matrix_vector_multiply

end module my_modules

! ------------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenACC

! Prompt: "Give me code for the following:
! Fortran, OpenACC, General matrix-vector multiply, subroutine"


! VERSION 1 !

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


! VERSION 2 !

module my_modules
  implicit none

contains

  subroutine my_matrix_vector_multiply(A, B, C, n) result(D)
    real(c_float), intent(in) :: A(n, n), B(n)
    real(c_float), intent(out) :: C(n)
    integer :: n

    ! Declare OpenACC variables
    integer :: tid, nthreads

    ! Set up OpenACC environment
    call acc_init(acc_device_type_host)
    call acc_set_num_threads(4)

    ! Offload computation to accelerator
    !$acc parallel num_gangs(4) gang_size(64) async(async_yes) present(A, B, C, n)

    ! Compute matrix-vector product
    do concurrent (i=1:n)
      do concurrent (j=1:n)
        C(i) = C(i) + A(i,j)*B(j)
      enddo
    enddo

    ! Synchronize threads
    !$acc synchronize

    ! Return
    return
  end subroutine my_matrix_vector_multiply

end module my_modules


! VERSION 3 !

module my_modules
  implicit none

contains

  subroutine my_matrix_vector_multiply(A, B, C, n) result(D)
    real(c_float), intent(in) :: A(n, n), B(n)
    real(c_float), intent(out) :: C(n)
    integer :: n

    ! Declare OpenACC variables
    integer :: tid, nthreads

    ! Set up OpenACC environment
    call acc_init(acc_device_type_host)
    call acc_set_num_threads(4)

    ! Offload computation to accelerator
    !$acc parallel num_gangs(4) gang_size(64) async(async_yes) present(A, B, C, n) copy(A, B, C)

    ! Compute matrix-vector product
    do concurrent (i=1:n)
      do concurrent (j=1:n)
        C(i) = C(i) + A(i,j)*B(j)
      enddo
    enddo

    ! Synchronize threads
    !$acc synchronize

    ! Return
    return
  end subroutine my_matrix_vector_multiply

end module my_modules
