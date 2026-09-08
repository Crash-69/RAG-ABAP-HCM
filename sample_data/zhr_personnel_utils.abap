*&---------------------------------------------------------------------*
*& Include ZHR_PERSONNEL_UTILS
*&---------------------------------------------------------------------*
REPORT zhr_personnel_utils.

TABLES: pa0001, pa0002.

FORM get_employee_name USING p_pernr TYPE persno
                       CHANGING c_name TYPE string.

  DATA: ls_pa0002 TYPE pa0002.

  SELECT SINGLE * FROM pa0002
    INTO ls_pa0002
    WHERE pernr = p_pernr
      AND endda = '99991231'.

  IF sy-subrc = 0.
    CONCATENATE ls_pa0002-vorna ls_pa0002-nachn INTO c_name SEPARATED BY space.
  ELSE.
    c_name = 'UNKNOWN'.
  ENDIF.

ENDFORM.

FORM get_employee_name USING p_pernr TYPE persno
                       CHANGING c_name TYPE string.

  DATA: ls_pa0002 TYPE pa0002.

  SELECT SINGLE * FROM pa0002
    INTO ls_pa0002
    WHERE pernr = p_pernr
      AND endda = '99991231'.

  IF sy-subrc = 0.
    CONCATENATE ls_pa0002-vorna ls_pa0002-nachn INTO c_name SEPARATED BY space.
  ELSE.
    c_name = 'UNKNOWN'.
  ENDIF.

ENDFORM.

CLASS lcl_hr_helper DEFINITION.
  PUBLIC SECTION.
    METHODS: get_org_unit
      IMPORTING iv_pernr TYPE persno
      RETURNING VALUE(rv_orgeh) TYPE orgeh.
ENDCLASS.

CLASS lcl_hr_helper IMPLEMENTATION.
  METHOD get_org_unit.
    DATA: ls_hrp1001 TYPE hrp1001.

    SELECT SINGLE * FROM hrp1001
      INTO ls_hrp1001
      WHERE objid = iv_pernr.

    IF sy-subrc = 0.
      rv_orgeh = ls_hrp1001-sobid.
    ENDIF.
  ENDMETHOD.
ENDCLASS.
