! OpenMP

! Prompt: "Give me code for the following:
! Fortran, OpenMP, AXPY, subroutine"

! VERSION 1 !

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


! VERSION 2 !

module matrix_ops
  implicit none

contains

  subroutine axpy(n, alpha, x, y) result(sum)
    integer :: n
    real(kind=8) :: alpha, x(n), y(n)
    real(kind=8) :: sum

    !$omp declare mapper(Preserve)
    !$omp parallel num_threads(4) default(none) shared(x,y)
    !$omp do concurrent(4)
    !$omp sections
    !$omp single

    sum = 0.d0
    do concurrent(4)
      do i = 1, n
        sum = sum + alpha * x(i) * y(i)
      enddo
    enddo

    !$omp end single
    !$omp endsections
    !$omp endparallel

  end subroutine axpy
end module matrix_ops

program main
  use matrix_ops
  implicit none

  integer :: n = 10
  real(kind=8) :: x(n), y(n), alpha = 3.d0

  call axpy(n, alpha, x, y)
  write (*,*) 'Sum = ', sum

end program main


! VERSION 3 !

subroutine axpy(n, alpha, x, y) result(sum)
  integer :: n
  real(kind=8) :: alpha, x(n), y(n)
  real(kind=8) :: sum

  !$omp declare mapper(Preserve)
  !$omp parallel num_threads(4) default(none) shared(x,y)
  !$omp do concurrent(4)
  !$omp sections
  !$omp single

  sum = 0.d0
  do concurrent(4)
    do i = 1, n
      sum = sum + alpha * x(i) * y(i)
    enddo
  enddo

  !$omp end single
  !$omp endsections
  !$omp endparallel

end subroutine axpy

program main
  implicit none

  integer :: n = 10
  real(kind=8) :: x(n), y(n), alpha = 3.d0

  call axpy(n, alpha, x, y)
  write (*,*) 'Sum = ', sum

end program main

! ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenMP (offload)

! Prompt: "Give me code for the following:
! Fortran, OpenMP (offload), AXPY, subroutine"

! VERSION 1 !

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


! VERSION 2 !

module matrix_ops
  implicit none

contains

  subroutine axpy(n, alpha, x, y) result(sum)
    integer :: n
    real(kind=8) :: alpha, x(n), y(n)
    real(kind=8) :: sum

    !$omp declare mapper(Preserve)
    !$omp parallel num_threads(4) default(none) shared(x,y)
    !$omp offload(4)
    !$omp do concurrent(4)
    !$omp sections
    !$omp single

    sum = 0.d0
    do concurrent(4)
      do i = 1, n
        sum = sum + alpha * x(i) * y(i)
      enddo
    enddo

    !$omp end single
    !$omp endsections
    !$omp endoffload
    !$omp endparallel

  end subroutine axpy
end module matrix_ops

program main
  use matrix_ops
  implicit none

  integer :: n = 10
  real(kind=8) :: x(n), y(n), alpha = 3.d0

  call axpy(n, alpha, x, y)
  write (*,*) 'Sum = ', sum

end program main


! VERSION 3 !

subroutine axpy(n, alpha, x, y) result(sum)
  integer :: n
  real(kind=8) :: alpha, x(n), y(n)
  real(kind=8) :: sum

  !$omp declare mapper(Preserve)
  !$omp parallel num_threads(4) default(none) shared(x,y)
  !$omp offload(4)
  !$omp do concurrent(4)
  !$omp sections
  !$omp single

  sum = 0.d0
  do concurrent(4)
    do i = 1, n
      sum = sum + alpha * x(i) * y(i)
    enddo
  enddo

  !$omp end single
  !$omp endsections
  !$omp endoffload
  !$omp endparallel

end subroutine axpy

program main
  implicit none

  integer :: n = 10
  real(kind=8) :: x(n), y(n), alpha = 3.d0

  call axpy(n, alpha, x, y)
  write (*,*) 'Sum = ', sum

end program main

! ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

! OpenACC

! Prompt: "Give me code for the following:
! Fortran, OpenACC, AXPY, subroutine"

! VERSION 1 !

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


! VERSION 2 !

module matrix_ops
  implicit none

contains

  subroutine axpy(n, alpha, x, y) result(sum)
    integer :: n
    real(kind=8) :: alpha, x(n), y(n)
    real(kind=8) :: sum

    !DECLARE_mapper( Preserve )
    !PARALLEL num_threads(4) default(none) shared(x,y)
    !ACCELERATOR( gpu )
    !DO CONCURRENT(4)
    !SECTIONS
    !SINGLE

    sum = 0.d0
    do concurrent(4)
      do i = 1, n
        sum = sum + alpha * x(i) * y(i)
      enddo
    enddo

    !END SINGLE
    !ENDIF defined(__GPU)
    !ENDACCELERATOR
    !ENDPARALLEL

  end subroutine axpy
end module matrix_ops

program main
  use matrix_ops
  implicit none

  integer :: n = 10
  real(kind=8) :: x(n), y(n), alpha = 3.d0

  call axpy(n, alpha, x, y)
  write (*,*) 'Sum = ', sum

end program main


! VERSION 3 !

subroutine axpy(n, alpha, x, y) result(sum)
  integer :: n
  real(kind=8) :: alpha, x(n), y(n)
  real(kind=8) :: sum

  !DECLARE_mapper( Preserve )
  !PARALLEL num_threads(4) default(none) shared(x,y)
  !ACCELERATOR( gpu )
  !DO CONCURRENT(4)
  !SECTIONS
  !SINGLE

  sum = 0.d0
  do concurrent(4)
    do i = 1, n
      sum = sum + alpha * x(i) * y(i)
    enddo
  enddo

  !END SINGLE
  !ENDIF defined(__GPU)
  !ENDACCELERATOR
  !ENDPARALLEL

end subroutine axpy

program main
  implicit none

  integer :: n = 10
  real(kind=8) :: x(n), y(n), alpha = 3.d0

  call axpy(n, alpha, x, y)
  write (*,*) 'Sum = ', sum

end program main

