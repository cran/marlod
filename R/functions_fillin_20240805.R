#' @return A numeric vector
#' @export
#'
#' @examples
#' Fillin()

Fillin=function(y, lod, n, tp, substitue)
{
  #############################################################
  # "impute.boot" function from "miWQS" package (version 0.4.4)
  #############################################################
  impute.boot=function(X, DL, Z = NULL, K = 5L, verbose = FALSE)
  {
    check_function.Lub=function(chemcol, dlcol, Z, K, verbose)
    {
      X <- check_X(as.matrix(chemcol))
      if (ncol(X) > 1) {
        warning("This approach cannot impute more than one chemical at time. Imputing first element...")
        chemcol <- chemcol[1]
      }
      else {
        chemcol <- as.vector(X)
      }
      if (is.null(dlcol))
        stop("dlcol is NULL. A detection limit is needed", call. = FALSE)
      if (is.na(dlcol))
        stop("The detection limit has missing values so chemical is not imputed.",
             immediate. = FALSE, call. = FALSE)
      if (!is.numeric(dlcol))
        stop("The detection limit is non-numeric. Detection limit must be numeric.",
             call. = FALSE)
      if (!is.vector(dlcol))
        stop("The detection limit must be a vector.", call. = FALSE)
      if (length(dlcol) > 1) {
        if (min(dlcol, na.rm = TRUE) == max(dlcol, na.rm = TRUE)) {
          dlcol <- unique(dlcol)
        }
        else {
          warning(" The detection limit is not unique, ranging from ",
                  min(dlcol, na.rm = TRUE), " to ", max(dlcol,
                                                        na.rm = TRUE), "; The smallest value is assumed to be detection limit",
                  call. = FALSE, immediate. = FALSE)
          detcol <- !is.na(chemcol)
          if (verbose) {
            cat("\n Summary when chemical is missing (BDL) \n")
            #print(summary(dlcol[detcol == 0]))
            cat("## when chemical is observed \n ")
            #print(summary(dlcol[detcol == 1]))
            dlcol <- min(dlcol, na.rm = TRUE)
          }
        }
      }
      if (is.null(Z)) {
        Z <- matrix(rep(1, length(chemcol)), ncol = 1)
      }
      if (anyNA(Z)) {
        warning("Missing covariates are ignored.")
      }
      if (!is.matrix(Z)) {
        if (is.vector(Z) | is.factor(Z)) {
          Z <- model.matrix(~., model.frame(~Z))[, -1, drop = FALSE]
        }
        else if (is.data.frame(Z)) {
          Z <- model.matrix(~., model.frame(~., data = Z))[,
                                                           -1, drop = FALSE]
        }
        else {
          stop("Please save Z as a vector, dataframe, or numeric matrix.")
        }
      }
      K <- check_constants(K)
      return(list(chemcol = chemcol, dlcol = dlcol, Z = Z, K = K))
    } #End of check_function.Lub

    impute.Lubin=function(chemcol, dlcol, Z = NULL, K = 5L, verbose = FALSE)
    {
      l <- check_function.Lub(chemcol, dlcol, Z, K, verbose)
      dlcol <- l$dlcol
      Z <- l$Z
      K <- l$K
      n <- length(chemcol)
      chemcol2 <- ifelse(is.na(chemcol), 1e-05, chemcol)
      event <- ifelse(is.na(chemcol), 3, 1)
      fullchem_data <- na.omit(data.frame(id = seq(1:n), chemcol2 = chemcol2,
                                          event = event, LB = rep(0, n), UB = rep(dlcol, n), Z))
      n <- nrow(fullchem_data)
      if (verbose) {
        cat("Dataset Used in Survival Model \n")
        #print(head(fullchem_data))
      }
      bootstrap_data <- matrix(0, nrow = n, ncol = K, dimnames = list(NULL,
                                                                      paste0("Imp.", 1:K)))
      beta_not <- NA
      std <- rep(0, K)
      unif_lower <- rep(0, K)
      unif_upper <- rep(0, K)
      imputed_values <- matrix(0, nrow = n, ncol = K, dimnames = list(NULL,
                                                                      paste0("Imp.", 1:K)))
      for (a in 1:K) {
        bootstrap_data[, a] <- as.vector(sample(1:n, replace = TRUE))
        data_sample <- fullchem_data[bootstrap_data[, a], ]
        freqs <- as.data.frame(table(bootstrap_data[, a]))
        freqs_ids <- as.numeric(as.character(freqs[, 1]))
        my.weights <- freqs[, 2]
        final <- fullchem_data[freqs_ids, ]
        my.surv.object <- survival::Surv(time = final$chemcol2,
                                         time2 = final$UB, event = final$event, type = "interval")
        model <- survival::survreg(my.surv.object ~ ., data = final[,
                                                                    -(1:5), drop = FALSE], weights = my.weights, dist = "lognormal",
                                   x = TRUE)
        beta_not <- rbind(beta_not, model$coefficients)
        std[a] <- model$scale
        Z <- model$x
        mu <- Z %*% beta_not[a + 1, ]
        unif_lower <- plnorm(1e-05, mu, std[a])
        unif_upper <- plnorm(dlcol, mu, std[a])
        u <- runif(n, unif_lower, unif_upper)
        imputed <- qlnorm(u, mu, std[a])
        imputed_values[, a] <- ifelse(fullchem_data$chemcol2 ==
                                        1e-05, imputed, chemcol)
      }
      x.miss.index <- ifelse(chemcol == 0 | is.na(chemcol), TRUE,
                             FALSE)
      indicator.miss <- sum(imputed_values[x.miss.index, ] > dlcol)
      if (verbose) {
        beta_not <- beta_not[-1, , drop = FALSE]
        cat("\n ## MLE Estimates \n")
        A <- round(cbind(beta_not, std), digits = 4)
        colnames(A) <- c(names(model$coefficients), "stdev")
        #print(A)
        cat("## Uniform Imputation Range \n")
        B <- rbind(format(range(unif_lower), digits = 3, scientific = TRUE),
                   format(range(unif_upper), digits = 3, nsmall = 3))
        rownames(B) <- c("unif_lower", "unif_upper")
        #print(B)
        cat("## Detection Limit:", unique(dlcol), "\n")
      }
      bootstrap_data <- apply(bootstrap_data, 2, as.factor)
      return(list(imputed_values = imputed_values, bootstrap_index = bootstrap_data,
                  indicator.miss = indicator.miss))
    } #End of impute.Lubin

    check_covariates=function(Z, X)
    {
      nZ <- if (is.vector(Z) | is.factor(Z)) {
        length(Z)
      }
      else {
        nrow(Z)
      }
      if (nrow(X) != nZ) {
        if (verbose) {
          cat("> X has", nrow(X), "individuals, but Z has", nZ,
              "individuals")
          stop("Can't Run. The total number of individuals with components X is different than total number of individuals with covariates Z.",
               call. = FALSE)
        }
      }
      if (anyNA(Z)) {
        p <- if (is.vector(Z) | is.factor(Z)) {
          1
        }
        else {
          ncol(Z)
        }
        index <- complete.cases(Z)
        Z <- Z[index, , drop = FALSE]
        X <- X[index, , drop = FALSE]
        dimnames(X)[[1]] <- 1:nrow(X)
        warning("Covariates are missing for individuals ", paste(which(!index),
                                                                 collapse = ", "), " and are ignored. Subjects renumbered in complete data.",
                call. = FALSE, immediate. = TRUE)
      }
      if (!is.null(Z) & !is.matrix(Z)) {
        if (is.vector(Z) | is.factor(Z)) {
          Z <- model.matrix(~., model.frame(~Z))[, -1, drop = FALSE]
        }
        else if (is.data.frame(Z)) {
          Z <- model.matrix(~., model.frame(~., data = Z))[,
                                                           -1, drop = FALSE]
        }
        else if (is.array(Z)) {
          stop("Please save Z as a vector, dataframe, or numeric matrix.")
        }
      }
      return(list(Z = Z, X = X))
    } #End of check_covariates

    is.wholenumber=function(x)
    {
      x > -1 && is.integer(x)
    } #End of is.wholenumber
    #is.wholenumber=function(x, tol = .Machine$double.eps^0.5)
    #{
    #  x > -1 && is.integer(x, tol)
    #} #End of is.wholenumber

    check_constants=function(K)
    {
      if (is.null(K))
        stop(sprintf(" must be non-null."), call. = TRUE)
      if (!is.numeric(K) | length(K) != 1)
        stop(sprintf(" must be numeric of length 1"), call. = TRUE)
      if (K <= 0)
        stop(sprintf(" must be a positive integer"), call. = TRUE)
      if (!is.wholenumber(K)) {
        K <- ceiling(K)
      }
      return(K)
    } #End of check_constants

    check_X=function(X)
    {
      if (is.null(X)) {
        stop("X is null. X must have some data.", call. = FALSE)
      }
      if (is.data.frame(X)) {
        X <- as.matrix(X)
      }
      if (!all(apply(X, 2, is.numeric))) {
        stop("X must be numeric. Some columns in X are not numeric.", call. = FALSE)
      }
      return(X)
    } #End of check_X

    check_imputation=function(X, DL, Z, K, T, n.burn, verbose = NULL)
    {
      if (is.vector(X)) {
        X <- as.matrix(X)
      }
      X <- check_X(X)
      if (!anyNA(X))
        stop("Matrix X has nothing to impute. No need to impute.", call. = FALSE)
      if (is.matrix(DL) | is.data.frame(DL)) {
        if (ncol(DL) == 1 | nrow(DL) == 1) {
          DL <- as.numeric(DL)
        }
        else {
          stop("The detection limit must be a vector.", call. = FALSE)
        }
      }
      stopifnot(is.numeric(DL))
      if (anyNA(DL)) {
        no.DL <- which(is.na(DL))
        warning("The detection limit for ", names(no.DL), " is missing.\n Both the chemical and detection limit is removed in imputation.",
                call. = FALSE)
        DL <- DL[-no.DL]
        if (!(names(no.DL) %in% colnames(X))) {
          if (verbose) {
            cat("  ##Missing DL does not match colnames(X) \n")
          }
        }
        X <- X[, -no.DL]
      }
      if (length(DL) != ncol(X)) {
        if (verbose) {
          cat("The following components \n")
        }
        X[1:3, ]
        if (verbose) {
          cat("have these detection limits \n")
        }
        DL
        stop("Each component must have its own detection limit.", call. = FALSE)
      }
      K <- check_constants(K)
      if (!is.null(n.burn) & !is.null(T)) {
        stopifnot(is.wholenumber(n.burn))
        T <- check_constants(T)
        if (!(n.burn < T))
          stop("Burn-in is too large; burn-in must be smaller than MCMC iterations T", call. = FALSE)
      }
      return(list(X = X, DL = DL, Z = Z, K = K, T = T))
    } #End of check_imputation

    check <- check_imputation(X = X, DL = DL, Z = Z, K = K, T = 5, n.burn = 4L, verbose)
    X <- check$X
    DL <- check$DL
    if (is.null(Z)) {
      Z <- matrix(1, nrow = nrow(X), ncol = 1, dimnames = list(NULL, "Intercept"))
    }
    else {
      l <- check_covariates(Z, X)
      X <- l$X
      Z <- l$Z
    }
    K <- check$K
    if (verbose) {
      cat("X \n")
      #print(summary(X))
      cat("DL \n")
      #print(DL)
      cat("Z \n")
      #print(summary(Z))
      cat("K =", K, "\n")
    }
    n <- nrow(X)
    c <- ncol(X)
    chemical.name <- colnames(X)
    results_Lubin <- array(dim = c(n, c, K), dimnames = list(NULL, chemical.name, paste0("Imp.", 1:K)))
    bootstrap_index <- array(dim = c(n, c, K), dimnames = list(NULL, chemical.name, paste0("Imp.", 1:K)))
    indicator.miss <- rep(NA, c)
    for (j in 1:c) {
      answer <- impute.Lubin(chemcol = X[, j], dlcol = DL[j], Z = Z, K = K)
      results_Lubin[, j, ] <- as.array(answer$imputed_values, dim = c(n, j, K))
      bootstrap_index[, j, ] <- array(answer$bootstrap_index, dim = c(n, 1, K))
      indicator.miss[j] <- answer$indicator.miss
    }
    total.miss <- sum(indicator.miss)
    if (total.miss == 0) {
      #message("#> Check: The total number of imputed values that are above the detection limit is ", sum(indicator.miss), ".")
    }
    else {
      #print(indicator.miss)
      stop("#> Error: Incorrect imputation is performed; some imputed values are above the detection limit. Could some of `X` have complete data under a lower bound (`DL`)?", call. = FALSE)
    }
    return(list(X.imputed = results_Lubin, bootstrap_index = answer$bootstrap_index, indicator.miss = total.miss))
  }
  ######################
  # End of "impute.boot"
  ######################

  #Substitution method using LOD
  if(substitue=="None") cen_y <- y

  #Substitution method using LOD
  if(substitue=="LOD") cen_y <- ifelse(y>=lod,y,lod)

  #Substitution method using LOD/2
  if(substitue=="LOD2") cen_y <- ifelse(y>=lod,y,lod/2)

  #Substitution method using LOD/square root of 2
  if(substitue=="LODS2") cen_y <- ifelse(y>=lod,y,lod/sqrt(2))

  #Beta-substitution method using mean
  if(substitue=="BetaMean"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)
    y_bar <- mean(lny1)
    z <- qnorm((length(y)-length(y1))/length(y))
    f_z <- dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    sy <- sqrt((y_bar-log((length(y)-length(y1))))^2/(dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))-qnorm((length(y)-length(y1))/length(y)))^2)
    f_sy_z <- (1-pnorm(qnorm((length(y)-length(y1))/length(y))-sy/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    beta_mean <- length(y)/(length(y)-length(y1))*pnorm(z-sy)*exp(-sy*z+(sy)^2/2)
    cen_y <- ifelse(y>=lod, y, lod*beta_mean)
  }
  #Beta-substitution method using geometric mean
  if(substitue=="BetaGM"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)
    y_bar <- mean(lny1)
    z <- qnorm((length(y)-length(y1))/length(y))
    f_z <- dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    sy <- sqrt((y_bar-log((length(y)-length(y1))))^2/(dnorm(qnorm((length(y)-length(y1))/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))-qnorm((length(y)-length(y1))/length(y)))^2)
    f_sy_z <- (1-pnorm(qnorm((length(y)-length(y1))/length(y))-sy/length(y)))/(1-pnorm(qnorm((length(y)-length(y1))/length(y))))
    beta_GM <- exp((-(length(y)-(length(y)-length(y1)))*length(y))/(length(y)-length(y1))*log(f_sy_z)-sy*z-(length(y)-(length(y)-length(y1)))/(2*(length(y)-length(y1))*length(y))*(sy)^2)
    cen_y <- ifelse(y>=lod, y, lod*beta_GM)
  }
  #Multiple imputation method without covariate
  #if(substitue=="MIWOCov"){
  #  y1 <- ifelse(y>=lod,y,NA)
  #  results_boot1 <- impute.boot(X=y1, DL=lod, Z=NULL, K=5, verbose = FALSE)
  #  cen_y <- apply(results_boot1$X.imputed, 1, mean)
  #}
  #Multiple imputation method with one covariate using id
  if(substitue=="MIWithID"){
    y1 <- ifelse(y>=lod,y,NA)
    results_boot2 <- impute.boot(X=y1, DL=lod, Z=rep(1:n, each=tp), K=5, verbose = TRUE)
    cen_y <- apply(results_boot2$X.imputed, 1, mean)
  }
  #Multiple imputation method with two covariates using id and visit
  if(substitue=="MIWithIDRM"){
    y1 <- ifelse(y>=lod,y,NA)
    results_boot3 <- impute.boot(X=y1, DL=lod, Z=cbind(rep(1:n, each=tp), rep(1:tp,n)), K=5, verbose = TRUE)
    cen_y <- apply(results_boot3$X.imputed, 1, mean)
  }
  #Multiple imputation method using QQ-plot approach
  if(substitue=="QQplot"){
    y1 <- y[which(y>=lod)]
    lny1 <- log(y1)

    obs <- rank(y)
    rank <- (obs-0.5)/length(y)
    zscore0 <- qnorm(rank)*ifelse(y>=lod,1,0)
    zscore1 <- zscore0[which(zscore0!=0)]

    data_frame0=data.frame(cbind(lny1,zscore1))
    beta_est=as.matrix(glm(lny1~zscore1, data=data_frame0, family=gaussian)$coefficients)
    lny_zscore <- beta_est[1,1] + beta_est[2,1]*qnorm(rank)
    y_zscore <- exp(lny_zscore)
    cen_y <-ifelse(y>=lod,y,y_zscore)
  }
  cen_y
}
