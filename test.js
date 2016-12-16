function createGrid(tableOrId,nbrPerPage,ajaxUrl,extraOptions) {
    var table = null;
    if(typeof tableOrId == "string") {
        table = $(tableOrId);
        }
    else {
        table = tableOrId;
        }
    if(ajaxUrl == undefined) {
        ajaxUrl = null;
        }
    var options = {
        bAutoWidth:false,
        bFilter:false,
        bLengthChange:false,
        bProcessing:true,
        bServerSide:ajaxUrl! = null,
        iDisplayLength:nbrPerPage,
        sAjaxSource:URL_SITE+ajaxUrl,
        sPaginationType:'full_numbers',
        sDom:'ipl<"block_spacer_5"><"clear"r>f<t>rip',
        oLanguage: {
            sProcessing:'Loading...',
            sEmptyTable:'No records to display.',
            sZeroRecords:'No records found.'
            },
        "fnDrawCallback":autoScrollUp
        };
    if(typeof extraOptions == "object") {
        for(var key in extraOptions) {
            options[key] = extraOptions[key];
            if(key == 'fnDrawCallback') {
                var callback = options[key];
                options[key] = function(o) {
                    autoScrollUp(o);
                    callback(o);
                    }
                }
            }
        }
    return table.dataTable(options);
    }