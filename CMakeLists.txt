# Set the name of the project and target:
SET(TARGET "elasticity")

SET(TARGET_SRC
    ${TARGET}.cpp
   )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.0.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
INCLUDE(${DEAL_II_FEATURE_CONFIG})

SET(CMAKE_C_COMPILER ${DEAL_II_MPI_C_COMPILER})
SET(CMAKE_CXX_COMPILER ${DEAL_II_MPI_CXX_COMPILER})

PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
