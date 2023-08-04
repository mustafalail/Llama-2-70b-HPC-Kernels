! OpenMP

! Prompt: "Give me code for the following:
! Fortran, OpenMP, AXPY, subroutine"

Program Example
  Implicit none

  Real :: a(10), b(10)
  Integer :: i

  !$omp parallel num_threads(4) default(none) shared(a,b)

  do i = 1, 10
    a(i) = 1.0d0
    b(i) = 2.0d0
  enddo

  call axpy(10, 3.0d0, a, 1, b, 1)

  !$omp barrier

  do i = 1, 10
    write (*,*) "Thread ", omp_get_thread_num(), ": a(", i, ") = ", a(i)
  enddo

contains

  subroutine axpy(n, alpha, x, incx, y, incy) result(dot)
    integer :: n, incx, incy
    real :: alpha, x(n), y(n)
    dot = 0.0d0
    !$omp parallel num_threads(4) default(none) shared(x,y)
    do i = 1, n
      dot = dot + alpha * x(i*incx) * y(i*incy)
    enddo
    !$omp barrier
    return
  end subroutine axpy
end program example

! ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenMP (offload)

! Prompt: "Give me code for the following:
! Fortran, OpenMP (offload), AXPY, subroutine"

Program Example
  Implicit none

  Real :: a(10), b(10)
  Integer :: i

  !$omp declare mapper(present(:))
  !$omp declare target(present(:))

  !$omp parallel num_threads(4) default(none) shared(a,b)

  do i = 1, 10
    a(i) = 1.0d0
    b(i) = 2.0d0
  enddo

  call axpy(10, 3.0d0, a, 1, b, 1)

  !$omp barrier

  do i = 1, 10
    write (*,*) "Thread ", omp_get_thread_num(), ": a(", i, ") = ", a(i)
  enddo

contains

  subroutine axpy(n, alpha, x, incx, y, incy) result(dot)
    integer :: n, incx, incy
    real :: alpha, x(n), y(n)
    dot = 0.0d0
    !$omp parallel num_threads(4) default(none) shared(x,y)
    do i = 1, n
      dot = dot + alpha * x(i*incx) * y(i*incy)
    enddo
    !$omp barrier
    return
  end subroutine axpy
end program example

! ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenACC

! Prompt: "Give me code for the following:
! Fortran, OpenACC, AXPY, subroutine"

Program Example
  Implicit none

  Real :: a(10), b(10)
  Integer :: i

  !declare acc variables
  real :: acc_a(10), acc_b(10)

  !set initial values
  do i = 1, 10
    a(i) = 1.0d0
    b(i) = 2.0d0
  enddo

  !copy data to accelerator
  call acc_copy_data(a, acc_a, 10)
  call acc_copy_data(b, acc_b, 10)

  !perform axpy on accelerator
  call axpy(10, 3.0d0, acc_a, 1, acc_b, 1)

  !copy result back to host
  call acc_copy_data(acc_a, a, 10)
  call acc_copy_data(acc_b, b, 10)

contains

  subroutine axpy(n, alpha, x, incx, y, incy) result(dot)
    integer :: n, incx, incy
    real :: alpha, x(n), y(n)
    dot = 0.0d0
    !$omp parallel num_threads(4) default(none) shared(x,y)
    do i = 1, n
      dot = dot + alpha * x(i*incx) * y(i*incy)
    enddo
    !$omp barrier
    return
  end subroutine axpy
end program example
