function [G, cvx_status] = solve_mvu_optimization(X, N)
    % SOLVE_MVU_OPTIMIZATION Solves the Maximum Variance Unfolding optimization problem
    % Inputs:
    % X: n-by-D data matrix (rows are points)
    % N: set of index pairs (i,j) in the neighborhood (e.g., from k-NN)

    n = size(X, 1);
    % Compute squared distance matrix without pdist2
    % D(i,j) = ||x_i - x_j||^2 = sum((x_i - x_j).^2)
    D = zeros(n, n);
    for i = 1:n
        for j = 1:n
            D(i,j) = sum((X(i,:) - X(j,:)).^2);
        end
    end

    % ==== Step 3: Solve MVU via CVX ====
    cvx_solver mosek
    cvx_begin sdp
        variable G(n, n) symmetric
        maximize( trace(G) )
        subject to
            G >= 0;
            sum(sum(G)) == 0;  % centering constraint

            % Distance-preserving constraints for neighbors
            for p = 1:size(N, 1)
                i = N(p, 1);
                j = N(p, 2);
                G(i,i) + G(j,j) - 2*G(i,j) == D(i,j);
            end
    cvx_end
end 