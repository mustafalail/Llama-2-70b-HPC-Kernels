!===================================================================================================================
!               OpenMP Programming Model
!===================================================================================================================

! Prompt: "Create 3 code suggestions using the following parameters: Fortran OpenMP General matrix-vector multiply subroutine."

! SUGGESTION 1 ! ----------------------------------

SUBROUTINE matrix_vector_multiply(A, x, y) result(y)
   real(kind=8), intent(in) :: A(n, n)
   real(kind=8), intent(in) :: x(n)
   real(kind=8), intent(out) :: y(n)
   integer :: i, j, n
   !$OMP PARALLEL DO default(none) shared(A, x, y)
   do concurrent (i = 1:n)
      do concurrent (j = 1:n)
         y(i) = y(i) + A(i, j) * x(j)
      enddo
   enddo
   !$OMP END PARALLEL DO
END SUBROUTINE matrix_vector_multiply


! SUGGESTION 2 ! ----------------------------------

SUBROUTINE matrix_vector_multiply(A, x, y) result(y)
   real(kind=8), intent(in) :: A(n, n)
   real(kind=8), intent(in) :: x(n)
   real(kind=8), intent(out) :: y(n)
   integer :: i, j, n
   !$OMP TASK LOOP DEFAULT(NONE) SHARED(A, X, Y)
   do task (i = 1:n)
      do task (j = 1:n)
         y(i) = y(i) + A(i, j) * x(j)
      enddo
   enddo
   !$OMP END TASK LOOP
END SUBROUTINE matrix_vector_multiply


! SUGGESTION 3 ! ----------------------------------

SUBROUTINE matrix_vector_multiply(A, x, y) result(y)
   real(kind=8), intent(in) :: A(n, n)
   real(kind=8), intent(in) :: x(n)
   real(kind=8), intent(out) :: y(n)
   integer :: i, j, n
   !$OMP PARALLEL REGION default(none) shared(A, x, y)
   do concurrent (i = 1:n)
      do concurrent (j = 1:n)
         y(i) = y(i) + A(i, j) * x(j)
      enddo
   enddo
   !$OMP END PARALLEL REGION
END SUBROUTINE matrix_vector_multiply



!===================================================================================================================
!               OpenMP(offload) Programming Model
!===================================================================================================================

! Prompt: "Create 3 code suggestions using the following parameters: Fortran OpenMP(offload) General matrix-vector multiply subroutine."

! SUGGESTION 1 ! ----------------------------------

SUBROUTINE matrix_vector_multiply(A, x, y) result(y)
   real(kind=8), intent(in) :: A(n, n)
   real(kind=8), intent(in) :: x(n)
   real(kind=8), intent(out) :: y(n)
   integer :: i, j, n
   !$OMP OFFLOAD DEVICE(dev) ENTIRE_FUNCTION
   !$OMP DECLARE mapper(present: dev)
   !$OMP SET_MAP(mapper, A, x, y)
   do concurrent (i = 1:n)
      do concurrent (j = 1:n)
         y(i) = y(i) + A(i, j) * x(j)
      enddo
   enddo
   !$OMP END OFFLOAD
END SUBROUTINE matrix_vector_multiply


! SUGGESTION 2 ! ----------------------------------

SUBROUTINE matrix_vector_multiply(A, x, y) result(y)
   real(kind=8), intent(in) :: A(n, n)
   real(kind=8), intent(in) :: x(n)
   real(kind=8), intent(out) :: y(n)
   integer :: i, j, n
   !$OMP OFFLOAD DEVICE(dev) ENTIRE_FUNCTION TEAMS(team)
   !$OMP DECLARE mapper(present: dev)
   !$OMP SET_MAP(mapper, A, x, y)
   !$OMP DATA SHARING(隊: team)
   do concurrent (i = 1:n)
      do concurrent (j = 1:n)
         y(i) = y(i) + A(i, j) * x(j)
      enddo
   enddo
   !$OMP END OFFLOAD
END SUBROUTINE matrix_vector_multiply


! SUGGESTION 3 ! ----------------------------------

SUBROUTINE matrix_vector_multiply(A, x, y) result(y)
   real(kind=8), intent(in) :: A(n, n)
   real(kind=8), intent(in) :: x(n)
   real(kind=8), intent(out) :: y(n)
   integer :: i, j, n
   !$OMP OFFLOAD DEVICE(dev) ENTIRE_FUNCTION SECTIONS(sect)
   !$OMP DECLARE mapper(present: dev)
   !$OMP SET_MAP(mapper, A, x, y)
   !$OMP DATA SHARING(sect: sect)
   do concurrent (i = 1:n)
      do concurrent (j = 1:n)
         y(i) = y(i) + A(i, j) * x(j)
      enddo
   enddo
   !$OMP END OFFLOAD
END SUBROUTINE matrix_vector_multiply



!===================================================================================================================
!               OpenACC Programming Model
!===================================================================================================================

! Prompt: "Create 3 code suggestions using the following parameters: Fortran OpenACC General matrix-vector multiply subroutine."

! SUGGESTION 1 ! ----------------------------------

SUBROUTINE matrix_vector_multiply(A, x, y) result(y)
   real(kind=8), intent(in) :: A(n, n)
   real(kind=8), intent(in) :: x(n)
   real(kind=8), intent(out) :: y(n)
   integer :: i, j, n
   !DECLARE acc_device *DEVICE(acc_device_t)
   !SET_DEVICE(acc_device, "GPU:0")
   !acc_enter_data(acc_device, A, x, y)
   do concurrent (i = 1:n)
      do concurrent (j = 1:n)
         y(i) = y(i) + A(i, j) * x(j)
      enddo
   enddo
   !acc_exit_data(acc_device)
END SUBROUTINE matrix_vector_multiply


! SUGGESTION 2 ! ----------------------------------

SUBROUTINE matrix_vector_multiply(A, x, y) result(y)
   real(kind=8), intent(in) :: A(n, n)
   real(kind=8), intent(in) :: x(n)
   real(kind=8), intent(out) :: y(n)
   integer :: i, j, n
   !DECLARE acc_device *DEVICE(acc_device_t)
   !SET_DEVICE(acc_device, "GPU:0")
   !acc_enter_data(acc_device, A, x, y)
   !$ACC KERNEL(gang, worker)
   do gang (i = 1:n)
      do worker (j = 1:n)
         y(i) = y(i) + A(i, j) * x(j)
      enddo
   enddo
   !acc_exit_data(acc_device)
END SUBROUTINE matrix_vector_multiply


! SUGGESTION 3 ! ----------------------------------

SUBROUTINE smvm(A, x, y) result(y)
   real(kind=8), intent(in) :: A(n, n)
   real(kind=8), intent(in) :: x(n)
   real(kind=8), intent(out) :: y(n)
   integer :: i, j, n
   !DECLARE acc_device *DEVICE(acc_device_t)
   !SET_DEVICE(acc_device, "GPU:0")
   !acc_enter_data(acc_device, A, x, y)
   !$ACC PARALLEL(num_teams, default(none))
   !$ACC LOOP VECTORIZATION(VECTORIZABLE)
   do concurrent (i = 1:n)
      do concurrent (j = 1:n)
         y(i) = y(i) + A(i, j) * x(j)
      enddo
   enddo
   !acc_exit_data(acc_device)
END SUBROUTINE smvm



! ---------------------------------------------------------------------------------------------------------------------------------------------------------
